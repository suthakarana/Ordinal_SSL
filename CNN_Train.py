import copy
import os
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchutils as tu
from oprounder import OptimizedRounder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.autograd import Variable
from torch_ema import ExponentialMovingAverage
from pytorch_lightning import seed_everything
from Models.DataLoader import *
from Models.DenseNet import MyDenseNet
from Models.efficientNet import *
from Models.ResNet import MyResNet
from Models.loss import *
from Models.loss import CE
from Models.util import *
import argparse
from sklearn.metrics import confusion_matrix
from Models.ViT import MyViT
from Models.MobileNetV2 import MyMobileNetV2
from Models.efficientNet import MyEfficientNet
from Models.ShuffleNetV2 import MyShuffleNetV2

class CNN_Train(nn.Module):
    def __init__(self, opt):
        super(CNN_Train, self).__init__()
        seed_everything(opt.seed, workers=True)
        self.opt = argparse.Namespace(**vars(opt))
        self.opt = opt

        base_output_dir = getattr(opt, "save_dir", getattr(opt, "result_dir", "./results"))
        run_name = getattr(opt, "save_name", None)
        if not run_name:
            run_name = (f"{getattr(opt, 'dataset', 'dataset')}_{getattr(opt, 'modelName', 'model')}_"
                        f"{getattr(opt, 'type', 'run')}_{getattr(opt, 'PL_TYPE', 'none')}_"
                        f"pL{getattr(opt, 'pL', 'na')}_seed{getattr(opt, 'seed', 'na')}")
        self.run_output_dir = os.path.join(str(base_output_dir), str(run_name))
        os.makedirs(self.run_output_dir, exist_ok=True)
        print(f"Run outputs: {self.run_output_dir}")

        self.temporal_mode = opt.temporal_mode
        self.use_parameter_ema = self.opt.temporal_mode in ("parameter_ema", "both")
        self.use_probability_ema = self.opt.temporal_mode in ("probability_ema", "both")
        print(
            f"Temporal mode: {self.opt.temporal_mode} | "f"Parameter EMA: {self.use_parameter_ema} | "f"Probability EMA: {self.use_probability_ema}")

        self.loader_L, self.loader_UL, self.loader_Test, self.valloader, cw, self.uniqueLbls = getDataLoaders(opt.dataset, opt.pL, opt.seed,
                                                                                                             16 if opt.type == "SSL" else opt.bs_l,
                                                                                                              opt.bs_u, opt.bs_Te, opt.bs_Val)

        self.labeled_iter = iter(self.loader_L)
        if self.loader_UL is not None:
            self.unlabeled_iter = iter(self.loader_UL)

        self.nClass = len(self.uniqueLbls)
        self.cw_L = torch.FloatTensor(cw).cuda(opt.gpuid)
        self.cw_U = self.cw_L.clone()

        self.net = self.loadModel()
        self.sm = nn.Softmax(dim=1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.995)

        # Count the primary network only. EMA is a shadow copy.
        self.total_params = int(
            sum(parameter.numel() for parameter in self.net.parameters())
        )
        print(f"Total parameters: {self.total_params:,}")

        self.optimized_rounder_MSE = OptimizedRounder(n_classes=self.nClass, metric="quadratic_kappa")
        self.rounder_is_fitted = False
        self.criterion_mse = nn.MSELoss().cuda(opt.gpuid)
        self.criterion = CE(self.nClass, opt.gpuid)

        self.clsWise_thr_w = torch.ones(self.nClass).cuda(opt.gpuid) * self.opt.init_thr
        self.clsWise_thr = torch.ones(self.nClass).cuda(opt.gpuid) * self.opt.init_thr

        num_u = len(self.loader_UL.dataset) if self.loader_UL is not None else 0
        self.prob_EMA = (torch.ones(num_u, self.nClass) / self.nClass).cuda(opt.gpuid)

        self.current_epoch = 0
        self.current_itr = 0

        n_iter_ssl_init = len(self.loader_UL) if self.loader_UL is not None else 1
        ramp_epochs = max(1, int(getattr(self.opt, "temporal_momentum_ramp_epochs", 1)))
        self.temporal_momentum_ramp_iters = max(1, ramp_epochs * n_iter_ssl_init)
        self.temporal_step = 0
        self.temporal_momentum = self.opt.momentum_Prob_start

        self.optimizer = optim.SGD(
            [
                {"params": self.net.backbone.parameters(), "weight_decay": 0.2 * opt.weight_decay},
                {"params": self.net.fc_ce.parameters(), "weight_decay": opt.weight_decay},
                {"params": self.net.fc_ce_w.parameters(), "weight_decay": opt.weight_decay},
                {"params": self.net.fc_mse.parameters(), "weight_decay": opt.weight_decay},
            ],
            lr=opt.lr,
            momentum=0.9,
            nesterov=True,
        )


        warmup_epochs = min(5, opt.fs_epoch) if opt.type != "FSL" else 5
        if opt.type == "FSL":
            warmup_steps = warmup_epochs * len(self.loader_L)
            total_steps = opt.n_epochs * len(self.loader_L)
        else:
            warmup_steps = opt.fs_epoch * len(self.loader_L)
            total_steps = opt.fs_epoch * len(self.loader_L) + \
                          (opt.n_epochs - opt.fs_epoch) * len(self.loader_UL)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )

    def loadModel(self):
        if self.opt.modelName in ("resnet18", "resnet50"):
            net = MyResNet(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes, model_name=self.opt.modelName)

        elif self.opt.modelName in ("vit_b_16", "vit_b_32", "vit_l_16"):
            net = MyViT(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes, model_name=self.opt.modelName,
                        pretrained=True, input_size=self.opt.input_size)

        elif self.opt.modelName == "mobilenet_v2":
            net = MyMobileNetV2(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes,
                                model_name=self.opt.modelName)

        elif self.opt.modelName in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2"):
            net = MyEfficientNet(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes,
                                 model_name=self.opt.modelName)

        elif self.opt.modelName == "shufflenet_v2_x2_0":
            net = MyShuffleNetV2(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes)

        elif self.opt.modelName == "densenet121":
            net = MyDenseNet(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes)

        else:
            raise ValueError(f"Invalid model: {self.opt.modelName}")

        return net.cuda(self.opt.gpuid)

    def getNextBatch_L(self):
        try:
            idx, Iw, Is, lbls = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.loader_L)
            idx, Iw, Is, lbls = next(self.labeled_iter)
        return idx.cuda(self.opt.gpuid), Iw.cuda(self.opt.gpuid), Is.cuda(self.opt.gpuid), lbls.cuda(self.opt.gpuid)

    def getNextBatch_UL(self):
        try:
            idx, Iw, Is, lbls = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.loader_UL)
            idx, Iw, Is, lbls = next(self.unlabeled_iter)
        return idx.cuda(self.opt.gpuid), Iw.cuda(self.opt.gpuid), Is.cuda(self.opt.gpuid), lbls.cuda(self.opt.gpuid)

    @torch.no_grad()
    def _temporal_momentum_update(self):
        self.temporal_step += 1
        progress = min(self.temporal_step / self.temporal_momentum_ramp_iters, 1.0)
        self.temporal_momentum = (
                    self.opt.momentum_Prob_start + progress * (self.opt.momentum_Prob - self.opt.momentum_Prob_start))

    # ----------------------------------------------------------------
    def get_class_distribution(self, labels):
        hist = torch.bincount(labels.long(), minlength=self.nClass).float()
        return hist / (hist.sum() + 1e-6)

    def get_pseudo_distribution(self, pseudo_labels, confidence):
        dist = torch.zeros(self.nClass).cuda(self.opt.gpuid)
        for c in range(self.nClass):
            mask = pseudo_labels == c
            if mask.sum() > 0:
                dist[c] = confidence[mask].sum()
        return dist / (dist.sum() + 1e-6)

    @torch.no_grad()
    def update_class_weights(self, y_l, pseudo_labels, selected_conf):
        if self.cw_L is None:
            return
        dist_l = self.get_class_distribution(y_l)
        dist_u = self.get_pseudo_distribution(pseudo_labels, selected_conf)

        combined_u = (1 - self.opt.lamda) * dist_l + self.opt.lamda * dist_u
        cw_u_new = 1.0 / (combined_u + 1e-6)
        cw_u_new = cw_u_new / cw_u_new.mean()
        cw_u_new = torch.clamp(cw_u_new, 0.3, 3.0)
        self.cw_U = (self.opt.cw_momentum * self.cw_U + (1 - self.opt.cw_momentum) * cw_u_new).detach()

    # ----------------------------------------------------------------

    @torch.no_grad()
    def updateConfEMA(self, out_ce, out_ce_w, idx):
        prob_ce, prob_ce_w = torch.softmax(out_ce, dim=1), torch.softmax(out_ce_w, dim=1)
        if self.opt.prob_source == "ce":
            prob_current = prob_ce
        elif self.opt.prob_source == "ce_w":
            prob_current = prob_ce_w
        elif self.opt.prob_source == "aggregate":
            prob_current = 0.5 * (prob_ce + prob_ce_w)
        else:
            raise ValueError(f"Unsupported prob_source: {self.prob_source}")
        if self.use_probability_ema:
            self.prob_EMA[idx] = self.temporal_momentum * self.prob_EMA[idx] + (
                        1.0 - self.temporal_momentum) * prob_current
        else:
            self.prob_EMA[idx] = prob_current
        self.prob_EMA[idx] = self.prob_EMA[idx] / self.prob_EMA[idx].sum(dim=1, keepdim=True).clamp_min(1e-6)

    @torch.no_grad()
    def avg_Prob_Thr(self, probs):
        conf, yp = torch.max(probs, dim=1)
        mask = conf >= self.clsWise_thr[yp]
        return yp.detach(), conf.detach(), mask.detach()

    @torch.no_grad()
    def distance_weighted_ordinal_concentration(self, probs):
        yp = torch.argmax(probs, dim=1)
        class_ids = torch.arange(self.nClass, device=probs.device, dtype=probs.dtype)
        expected = (probs * class_ids).sum(dim=1, keepdim=True)
        distances = torch.abs(class_ids.unsqueeze(0) - expected)
        weights = torch.exp(-distances / self.opt.ord_tau)
        conf = torch.sum(probs * weights, dim=1)
        mask = conf >= self.clsWise_thr[yp]
        return yp.detach(), conf.detach(), mask.detach()

    @torch.no_grad()
    def updateThreshold(self, conf, pseudo_labels):
        if conf is None or conf.numel() == 0:
            return

        if self.opt.thr_mode == "global_fixed":
            self.clsWise_thr.fill_(self.opt.init_thr)
            return

        if self.opt.thr_mode == "global_adaptive":
            batch_thr = torch.full((self.nClass,), conf.mean(), device=self.clsWise_thr.device)

        elif self.opt.thr_mode == "classwise_adaptive":
            batch_thr = self.clsWise_thr.clone()
            for c in range(self.nClass):
                idx = pseudo_labels == c
                if idx.any():
                    batch_thr[c] = conf[idx].mean()
        else:
            raise ValueError(f"Unsupported thr_mode: {self.opt.thr_mode}")

        batch_thr = batch_thr.clamp(min=self.opt.minThrClamp, max=self.opt.maxThrClamp)
        m = self.opt.momentum_thr
        self.clsWise_thr = (m * self.clsWise_thr + (1.0 - m) * batch_thr).detach()

    def getLoss(self, out_ce, out_ce_w, out_mse, y, cw=None, mask=None):
        ce_loss = self.criterion.CE(out_ce, y, mask=mask)
        ce_loss_w = self.criterion.CE(out_ce_w, y, cw, mask=mask)
        if mask is not None and mask.sum() > 0:
            mse_elem = F.mse_loss(out_mse[mask], y[mask].view(-1, 1).float(), reduction="none").squeeze(1)
            mse_loss = mse_elem.mean()
        else:
            mse_loss = self.criterion_mse(out_mse, y.view(-1, 1).float())
        return self.opt.w_ce * ce_loss + self.opt.w_ce_w * ce_loss_w + self.opt.w_mse * mse_loss

    def train_SSL(self, n_iter_oneepoch, ssl_epoch_counter):
        self.net.train()

        loss_tot = 0.0
        loss_sup_tot, loss_ssl_tot = 0.0, 0.0

        mask_all, conf_all = [], []
        y_l_all, y_u_all, y_up_all = [], [], []
        pred_rounder_mse_all, gt_rounder_all = [], []

        pl_fn = {'AvgProb': self.avg_Prob_Thr, 'OrdDist': self.distance_weighted_ordinal_concentration}.get(
            self.opt.PL_TYPE)

        if pl_fn is None:
            raise ValueError(f"Invalid PL_TYPE '{self.opt.PL_TYPE}'.")

        print()

        for i in range(n_iter_oneepoch):
            if self.use_probability_ema:
                self._temporal_momentum_update()
            idx_l, _, Is_l, y_l = self.getNextBatch_L()
            idx_u, Iw_u, Is_u, y_u = self.getNextBatch_UL()

            bs_l = Is_l.size(0)
            bs_u = Is_u.size(0)

            mask, conf = None, None

            if self.opt.w_pl > 0:
                out_ce_t, out_ce_w_t, _ = self.testImage(Iw_u)
                self.updateConfEMA(out_ce_t, out_ce_w_t, idx_u)
                y_u_pseudo, conf, mask = pl_fn(self.prob_EMA[idx_u])

            # y_u_pseudo = y_u.clone()
            # mask = torch.ones_like(y_u).bool()
            # conf = torch.ones_like(y_u).float()

            out_ce, out_ce_w, out_mse = self.net(torch.cat([Is_l, Is_u], dim=0))

            cw_sup = self.get_supervised_class_weights()
            loss_sup = self.getLoss(out_ce[:bs_l], out_ce_w[:bs_l], out_mse[:bs_l], y_l, cw=cw_sup)
            loss = loss_sup
            loss_ssl = torch.tensor(0.0, device=y_l.device)
            loss_sup_tot += loss_sup.item()

            if self.opt.w_pl > 0 and mask is not None and mask.any():
                cw_u = self.get_unlabeled_class_weights()
                loss_ssl = self.getLoss(out_ce[bs_l:], out_ce_w[bs_l:], out_mse[bs_l:], y_u_pseudo, cw=cw_u, mask=mask)
                loss += self.opt.w_pl * loss_ssl
                pred_rounder_mse_all.append(out_mse[bs_l:][mask].detach())
                gt_rounder_all.append(y_u_pseudo[mask].detach())
            loss_ssl_tot += loss_ssl.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.use_parameter_ema:
                self.ema.update()

            self.current_itr += 1
            loss_tot += loss.item()

            y_l_all.append(y_l.detach())
            y_u_all.append(y_u.detach())

            if self.opt.w_pl > 0:
                y_up_all.append(y_u_pseudo.detach())
                mask_all.append(mask.detach())
                conf_all.append(conf.detach())

            mask_n = mask.sum().item() if mask is not None else 0
            conf_mean = conf.mean().item() if conf is not None else 0.0
            print(f"\rSSL Train: {i + 1:3d}/{n_iter_oneepoch} | loss {loss.item():.4f} | "
                  f"mask {mask_n:4d}/{bs_u} | conf {conf_mean:.3f}", end="")
        print("\n")
        y_l_all = torch.cat(y_l_all)
        y_u_all = torch.cat(y_u_all)
        ns, accPseudoLbls, bacc = 0, 0, 0
        if self.opt.w_pl > 0:
            y_up_all = torch.cat(y_up_all)
            mask_all = torch.cat(mask_all)
            conf_all = torch.cat(conf_all)
            ns = mask_all.sum().item()
            if ns > 0:
                selected_mask = mask_all.bool()
                selected_gt = y_u_all[selected_mask]
                selected_pl = y_up_all[selected_mask]
                selected_conf = conf_all[selected_mask]
                accPseudoLbls = accuracy_score(selected_gt.cpu().numpy(), selected_pl.cpu().numpy())
                bacc = balanced_accuracy_score(selected_gt.cpu().numpy(), selected_pl.cpu().numpy())
                thr_used = self.clsWise_thr.detach().clone()
                self.printStatPL(y_u_all, y_up_all, conf_all, mask_all, thr_used)
                if self.opt.cw_mode == "dynamic":
                    self.update_class_weights(y_l_all.long(), selected_pl, selected_conf)
                self.updateThreshold(conf_all, y_up_all)
            print(f"SSL summary | n={ns} | acc_pl={accPseudoLbls:.3f} | bacc_pl={bacc:.3f}")
        if len(gt_rounder_all) > 0 and (self.current_epoch % 5 == 0 or self.current_epoch < self.opt.fs_epoch):
            gt_all = torch.cat(gt_rounder_all)
            pred_all = torch.cat(pred_rounder_mse_all)
            self.optimized_rounder_MSE.fit(pred_all.view(-1, 1).cpu().numpy(), gt_all.view(-1, 1).float().cpu().numpy())
            self.rounder_is_fitted = True

        return loss_tot, loss_sup_tot, loss_ssl_tot, ns, accPseudoLbls, bacc

    def train_FS(self, n_iter_oneepoch):
        self.net.train()
        pred_all_mse, pred_all_ce, pred_all_ce_w, gt_all = [], [], [], []
        loss_tot = 0

        for i, (_, _, I, y) in enumerate(self.loader_L):
            I = Variable(I.cuda(self.opt.gpuid))
            y = Variable(y.cuda(self.opt.gpuid))

            out_ce, out_ce_w, out_mse = self.net(I)

            cw_sup = self.get_supervised_class_weights()
            loss = self.getLoss(out_ce, out_ce_w, out_mse, y, cw=cw_sup)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            if self.use_parameter_ema:
                self.ema.update()
            self.current_itr += 1

            loss_tot += loss.item()
            gt_all.append(y.detach())
            pred_all_ce.append(out_ce.detach())
            pred_all_ce_w.append(out_ce_w.detach())
            pred_all_mse.append(out_mse.detach())
            print(f"\rTrain: {i + 1}/{n_iter_oneepoch}", end="")

        gt_all = torch.cat(gt_all, 0)
        pred_all_ce = self.sm(torch.cat(pred_all_ce, 0))
        pred_all_ce_w = self.sm(torch.cat(pred_all_ce_w, 0))
        pred_all_mse = torch.cat(pred_all_mse, 0)

        self.optimized_rounder_MSE.fit(pred_all_mse.view(-1, 1).cpu().numpy(), gt_all.view(-1, 1).float().cpu().numpy())
        self.rounder_is_fitted = True
        pl_MSE = self.optimized_rounder_MSE.predict(pred_all_mse.cpu().numpy())

        re_mse = getScores(gt_all.cpu().numpy(), pl_MSE)
        re_ce = getScore_fromPred(gt_all.cpu().numpy(), pred_all_ce)
        re_ce_w = getScore_fromPred(gt_all.cpu().numpy(), pred_all_ce_w)

        idx = 2
        return loss_tot, re_mse[idx], re_ce[idx], re_ce_w[idx]

    @torch.no_grad()
    def testImage(self, I):
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            if self.use_parameter_ema:
                with self.ema.average_parameters():
                    out_ce, out_ce_w, out_mse = self.net(I)
            else:
                out_ce, out_ce_w, out_mse = self.net(I)
        if was_training:
            self.net.train()
        return out_ce, out_ce_w, out_mse

    @torch.no_grad()
    def test(self, loader, desc="Eval"):
        self.net.eval()
        pred_mse, pred_ce, pred_ce_w, gt = [], [], [], []
        total = len(loader)

        for i, (I, y) in enumerate(loader):
            I = Variable(I.cuda(self.opt.gpuid))
            y = Variable(y.cuda(self.opt.gpuid))

            out_ce, out_ce_w, out_mse = self.testImage(I)
            gt.append(y.detach())
            pred_ce.append(out_ce.detach())
            pred_ce_w.append(out_ce_w.detach())
            pred_mse.append(out_mse.detach())
            print(f"\r{desc}: {i + 1}/{total}", end="")

        print('\t:\t', end='')
        preds = {"gt": torch.cat(gt, 0).cpu(), "mse": torch.cat(pred_mse, 0).cpu(),
                 "ce": self.sm(torch.cat(pred_ce, 0)).cpu(),
                 "ce_w": self.sm(torch.cat(pred_ce_w, 0)).cpu()}
        return preds

    def score_predictions(self, preds, refit=False):
        gt_np = preds["gt"].view(-1).numpy()
        mse_np = preds["mse"].numpy()

        if refit:
            self.optimized_rounder_MSE.fit(mse_np, gt_np.reshape(-1, 1))
            self.rounder_is_fitted = True

        mse_rounded = np.asarray(self.optimized_rounder_MSE.predict(mse_np)).astype(int).reshape(-1)
        re_mse = getScores(gt_np, mse_rounded, probs=None, num_classes=self.nClass)
        re_ce = getScore_fromPred(gt_np, preds["ce"])
        re_ce_w = getScore_fromPred(gt_np, preds["ce_w"])

        return {"CE": re_ce, "CE_w": re_ce_w, "MSE": re_mse}

    def _make_checkpoint(self):
        return {
            "model": copy.deepcopy(self.net.state_dict()),
            "ema": copy.deepcopy(self.ema.state_dict()),
            "rounder": copy.deepcopy(self.optimized_rounder_MSE),
            "clsWise_thr": self.clsWise_thr.clone(),
            "clsWise_thr_w": self.clsWise_thr_w.clone(),
            "cw_U": self.cw_U.clone(),
            "cw_L": self.cw_L.clone(),
            "current_itr": self.current_itr,
            "current_epoch": self.current_epoch,
        }

    def _load_checkpoint(self, checkpoint):
        self.net.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema"])
        if "rounder" in checkpoint:
            self.optimized_rounder_MSE = copy.deepcopy(checkpoint["rounder"])
            self.rounder_is_fitted = True
        if "clsWise_thr" in checkpoint:
            self.clsWise_thr = checkpoint["clsWise_thr"].clone()
        if "clsWise_thr_w" in checkpoint:
            self.clsWise_thr_w = checkpoint["clsWise_thr_w"].clone()
        if "cw_U" in checkpoint:
            self.cw_U = checkpoint["cw_U"].clone()
        if "cw_L" in checkpoint:
            self.cw_L = checkpoint["cw_L"].clone()
        if "current_itr" in checkpoint:
            self.current_itr = checkpoint["current_itr"]
        if "current_epoch" in checkpoint:
            self.current_epoch = checkpoint["current_epoch"]

    def get_supervised_class_weights(self):
        if self.opt.cw_mode == "none":
            return None
        return self.cw_L

    def get_unlabeled_class_weights(self):
        if self.opt.cw_mode == "none":
            return None
        if self.opt.cw_mode == "fixed":
            return self.cw_L
        if self.opt.cw_mode == "dynamic":
            return self.cw_U.detach()
        raise ValueError(f"Unsupported cw_mode: {self.opt.cw_mode}")

    @staticmethod
    def _metric_text(value):
        return "--" if isinstance(value, (float, np.floating)) and not np.isfinite(value) else f"{float(value):.4f}"

    def _print_eval_metrics(self, name, results):
        headers = ["Branch", "Acc", "BAcc", "QWK", "F1micro", "F1macro", "AUC", "Prec", "Recall", "Sens", "MCC", "Spec"]
        widths = [8] + [10] * 11
        table_width = sum(widths)
        print("\n" + "=" * table_width)
        print(name)
        print("=" * table_width)
        print("".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
        print("-" * table_width)
        for branch in ["MSE", "CE", "CE_w"]:
            row = f"{branch:<{widths[0]}}"
            for value, width in zip(results[branch], widths[1:]):
                row += f"{self._metric_text(value):<{width}}"
            print(row)
        print("=" * table_width)

    def _ce_classwise_predictions(self, preds):
        gt_np = preds["gt"].view(-1).numpy()
        ce_probs = preds["ce"].numpy()
        ce_pred = np.argmax(ce_probs, axis=1)
        return getClasswiseScores(gt_np,ce_pred,probs=ce_probs,num_classes=self.nClass,class_names=list(self.uniqueLbls))

    def _print_ce_classwise_metrics(self, name, classwise_rows):
        headers = ["Class","Support","Pred","Acc","BAcc", "F1micro","F1macro", "AUC","Prec", "Recall","Sens","MCC","Spec"]
        widths = [12, 9, 8] + [10] * 10
        table_width = sum(widths)
        metric_keys = ["accuracy", "balanced_accuracy","f1_micro","f1_macro","auc", "precision","recall","sensitivity","mcc", "specificity"]
        print("\n" + "=" * table_width)
        print(f"{name} - CE Classwise Summary")
        print("=" * table_width)
        print(
            "".join(
                f"{header:<{width}}"
                for header, width in zip(headers, widths)
            )
        )
        print("-" * table_width)
        for item in classwise_rows:
            row = (
                f"{str(item['class_name']):<{widths[0]}}"
                f"{item['support']:<{widths[1]}}"
                f"{item['predicted']:<{widths[2]}}"
            )
            for key, width in zip(metric_keys, widths[3:]):
                row += f"{self._metric_text(item[key]):<{width}}"
            print(row)
        print("=" * table_width)

    def _save_final_metric_summaries(self,overall,ce_classwise):
        overall_rows = []
        for split in ["Validation", "Test"]:
            for branch in ["MSE", "CE", "CE_w"]:
                row = {
                    "split": split,
                    "branch": branch,
                }
                row.update(
                    {
                        name: value
                        for name, value in zip(
                        OVERALL_METRIC_NAMES,
                        overall[split][branch],
                    )
                    }
                )
                overall_rows.append(row)
        ce_classwise_rows = []
        for split in ["Validation", "Test"]:
            for item in ce_classwise[split]:
                ce_classwise_rows.append(
                    {
                        "split": split,
                        "branch": "CE",
                        **item,
                    }
                )
        overall_path = os.path.join(
            self.run_output_dir,
            "final_overall_metrics.csv",
        )
        classwise_path = os.path.join(
            self.run_output_dir,
            "final_CE_classwise_metrics.csv",
        )
        save_rows_to_csv(overall_rows, overall_path)
        save_rows_to_csv(ce_classwise_rows, classwise_path)
        print(f"Overall metrics saved: {overall_path}")
        print(f"CE classwise metrics saved: {classwise_path}")

    def _save_test_predictions(self, preds):
        gt=preds["gt"].view(-1).numpy().astype(np.int64)
        prob=preds["ce"].numpy().astype(np.float32)
        pred=np.argmax(prob,axis=1).astype(np.int64)
        path=os.path.join(self.run_output_dir,"test_predictions.npz")
        np.savez_compressed(path,sample_id=np.arange(len(gt),dtype=np.int64),
                            y_true=gt,
                            y_pred=pred,
                            y_prob=prob,
                            algorithm=np.asarray("OrdDist"),
                            seed=np.asarray(getattr(self.opt,"seed",-1)),
                            branch=np.asarray("CE"))
        print(f"Test predictions saved: {path}")

    def _save_test_confusion_matrix(self, preds):
        gt = preds["gt"].view(-1).numpy().astype(np.int64)
        ce_prob = preds["ce"].numpy()
        ce_pred = np.argmax(ce_prob, axis=1).astype(np.int64)
        labels = np.arange(self.nClass)
        cm = confusion_matrix(gt, ce_pred, labels=labels)
        row_totals = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm.astype(np.float64),row_totals,out=np.zeros_like(cm, dtype=np.float64),where=row_totals != 0,)
        class_names = [str(name) for name in self.uniqueLbls]
        header = "true/pred," + ",".join(class_names)
        cm_path = os.path.join( self.run_output_dir,"final_test_CE_confusion_matrix.csv")
        cm_normalized_path = os.path.join(self.run_output_dir,"final_test_CE_confusion_matrix_normalized.csv")
        np.savetxt(cm_path,cm,delimiter=",",fmt="%d",header=header,comments="")
        np.savetxt(cm_normalized_path,cm_normalized, delimiter=",",fmt="%.6f",header=header,comments="")
        print("\n" + "=" * 72)
        print("FINAL TEST CONFUSION MATRIX - CE BRANCH")
        print("Rows: true classes | Columns: predicted classes")
        print("=" * 72)
        print("Class order:", class_names)
        print(cm)
        print("\nRow-normalized confusion matrix:")
        print(np.round(cm_normalized, 4))
        print("=" * 72)
        print(f"Confusion matrix saved: {cm_path}")
        print(f"Normalized confusion matrix saved: {cm_normalized_path}")

    def final_evaluation(self):
        print("\n=== Final Evaluation ===")

        val_preds = self.test(self.valloader, desc="Validation")
        val_overall = self.score_predictions(val_preds, refit=False)
        val_ce_classwise = self._ce_classwise_predictions(val_preds)

        test_preds = self.test(self.loader_Test, desc="Test")
        self._save_test_predictions(test_preds)
        self._save_test_confusion_matrix(test_preds)

        test_overall = self.score_predictions(test_preds, refit=False)
        test_ce_classwise = self._ce_classwise_predictions(test_preds)

        self._print_eval_metrics("Validation", val_overall)
        self._print_ce_classwise_metrics("Validation", val_ce_classwise)

        self._print_eval_metrics("Test", test_overall, )
        self._print_ce_classwise_metrics("Test", test_ce_classwise)

        self._save_final_metric_summaries(overall={"Validation": val_overall, "Test": test_overall},
                                          ce_classwise={"Validation": val_ce_classwise,"Test": test_ce_classwise})
        return test_overall

    def fit_rounder(self, preds):
        self.optimized_rounder_MSE.fit(preds["mse"].numpy(), preds["gt"].view(-1).numpy())
        self.rounder_is_fitted = True

    @torch.no_grad()
    def printStatPL(self, y_true, y_pred, conf, mask, thr_used):
        mask = mask.bool()
        print( "lbl	 total	 selected	 coverage	 predicted	 precision	 recall	 selAcc	 thr	 maxConf	 cw")
        for c in range(self.nClass):
            true_c = y_true == c
            pred_c = y_pred == c
            selected_true_c = mask & true_c
            selected_pred_c = mask & pred_c
            correct_c = mask & true_c & pred_c
            total = true_c.sum().item()
            selected = selected_true_c.sum().item()
            predicted = selected_pred_c.sum().item()
            correct = correct_c.sum().item()
            coverage = selected / total if total > 0 else 0.0
            precision = correct / predicted if predicted > 0 else 0.0
            recall = correct / total if total > 0 else 0.0
            selected_accuracy = correct / selected if selected > 0 else 0.0
            max_conf = conf[true_c].max().item() if true_c.any() else 0.0
            print(f"{c:1d}	{total:6d}	{selected:8d}	{coverage * 100:7.2f}%	"
                  f"{predicted:9d}	{precision * 100:8.2f}%	{recall * 100:7.2f}%	"
                  f"{selected_accuracy * 100:7.2f}%	{thr_used[c].item():.3f}	"
                  f"{max_conf:.3f}	{self.cw_U[c].item():.3f}")

    def printScore(self, re, ends):
        print("%2.2f\t%2.2f\t%.3f\t%.3f" % (re[0], re[1], re[2], re[3]), end=ends)

    def _cuda_synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.opt.gpuid)

    def _save_time_and_parameters(self, total_seconds):
        total_seconds_int = max(0, int(round(total_seconds)))
        hours, remainder = divmod(total_seconds_int, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_hms = (
            f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        )

        output_path = os.path.join(
            self.run_output_dir,
            "time_and_parameters.csv",
        )
        save_rows_to_csv(
            [
                {
                    "total_time_seconds": float(total_seconds),
                    "total_time_hms": total_time_hms,
                    "total_parameters": int(self.total_params),
                }
            ],
            output_path,
        )

        print("\n=== Computational Summary ===")
        print(f"Total time       : {total_time_hms} ({total_seconds:.2f} s)")
        print(f"Total parameters : {self.total_params:,}")
        print(f"Saved            : {output_path}")

    def iterate_CNN(self):
        n_labeled = len(self.loader_L.dataset)
        n_unlabeled = len(self.loader_UL.dataset) if self.loader_UL is not None else 0
        n_iter_fsl = len(self.loader_L)
        n_iter_ssl = len(self.loader_UL) if self.loader_UL is not None else 0
        total_epochs = self.opt.n_epochs
        results = []
        self._cuda_synchronize()
        start_time = time.perf_counter()
        patience_counter = 0
        best_Kappa = -np.inf
        best_checkpoint = None
        ssl_started = False
        ssl_epoch_counter = 0
        for epoch in range(total_epochs):
            self.current_epoch = epoch
            ns = 0
            accPseudoLbls = 0
            bacc = 0
            kappa_mse = 0.0
            kappa_ce = 0.0
            kappa_ce_w = 0.0
            is_fsl_phase = (self.opt.type == "FSL") or (epoch < self.opt.fs_epoch)
            if is_fsl_phase:
                train_loss, kappa_mse, kappa_ce, kappa_ce_w = self.train_FS(n_iter_fsl)
                is_warmup = True
            else:
                if not ssl_started:
                    ssl_started = True
                    patience_counter = 0
                    self.loader_L = torch.utils.data.DataLoader(self.loader_L.dataset, batch_size=self.opt.bs_l,
                                                                shuffle=True, num_workers=self.loader_L.num_workers)
                    self.labeled_iter = iter(self.loader_L)
                    print( f"\n>>> Warmup complete. Starting SSL training | bs_l={self.opt.bs_l} | bs_u={self.opt.bs_u}\n")
                ssl_epoch_counter += 1
                train_loss, loss_sup, loss_ssl, ns, accPseudoLbls, bacc = self.train_SSL(n_iter_ssl,ssl_epoch_counter=ssl_epoch_counter)
                is_warmup = False

            lr = tu.get_lr(self.optimizer)
            preds = self.test(self.valloader, desc="Validation")
            reEval = self.score_predictions(preds, refit=True)

            current_kappa = reEval["MSE"][2]
            if current_kappa > best_Kappa + 1e-4:
                best_Kappa = current_kappa
                best_checkpoint = self._make_checkpoint()
                patience_counter = 0

                print(f"\n>>> New best Validation QWK : {best_Kappa:.4f}")
            else:

                if self.opt.type == "FSL" or ssl_epoch_counter > self.opt.patience_warmup:
                    patience_counter += 1
            print(f"\nEpoch {epoch + 1:3d}/{total_epochs} | LR:{lr:.6f} | Loss:{train_loss:.4f}", end="")
            if is_warmup:
                print(f" | Train QWK -> MSE:{kappa_mse:.4f} CE:{kappa_ce:.4f} CE-W:{kappa_ce_w:.4f}")
            else:
                ratio = loss_ssl / max(loss_sup, 1e-8)
                print(
                    f" | SSL | Lsup:{loss_sup:.3f} Lssl:{loss_ssl:.3f} R:{ratio:.2f} | PC:{patience_counter}/{self.opt.patience} | SSL_ep:{ssl_epoch_counter}")
            self._print_eval_metrics("Validation", reEval)
            if patience_counter >= self.opt.patience:
                print("\nEarly stopping triggered")
                break
            re = [lr, train_loss]
            if is_warmup:
                re.extend([kappa_mse, kappa_ce, kappa_ce_w])
            else:
                re.extend([ns, accPseudoLbls, bacc])

            re.extend(reEval["MSE"])
            re.extend(reEval["CE"])
            re.extend(reEval["CE_w"])
            results.append(re)

        if best_checkpoint is not None:
            print("\nLoading best checkpoint...")
            self._load_checkpoint(best_checkpoint)
        else:
            print("\nWarning: No best checkpoint found.")
        self.best_validation_qwk = float(best_Kappa)
        final_results = self.final_evaluation()

        final_vec = np.concatenate([
            np.asarray(final_results["MSE"], dtype=np.float32),
            np.asarray(final_results["CE"], dtype=np.float32),
            np.asarray(final_results["CE_w"], dtype=np.float32)
        ])

        self._cuda_synchronize()
        total_run_seconds = time.perf_counter() - start_time
        self._save_time_and_parameters(total_run_seconds)

        return final_vec, self.net
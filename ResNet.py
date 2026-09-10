import torch
import torch.nn as nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, out_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_in, out_dim)
        )
    def forward(self, x):
        return self.layers(x)


class MyResNet(nn.Module):
    def __init__(self, num_classes=5, useBothEyes=False, model_name="resnet18"):
        super().__init__()

        self.useBothEyes = useBothEyes
        self.model_name = model_name
        if model_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            nfea = resnet.fc.in_features  # 512
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        elif model_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            nfea = resnet.fc.in_features   # 2048
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        else:
            raise ValueError(f"Unsupported model_name for MyResNet: {model_name}")

        times = 2 if useBothEyes else 1
        feature_dim = nfea * times

        self.dout = nn.Dropout(p=0.2)
        self.mse_feat = MLP(feature_dim, feature_dim, dropout=0.1)
        self.ce_feat = MLP(feature_dim, feature_dim, dropout=0.1)
        self.ce_w_feat = MLP(feature_dim, feature_dim, dropout=0.1)

        self.fc_mse = nn.Linear(feature_dim, 1)
        self.fc_ce = nn.Linear(feature_dim, num_classes)
        self.fc_ce_w = nn.Linear(feature_dim, num_classes)

    def extract(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2=None):
        x1 = self.extract(x1)

        if self.useBothEyes and x2 is not None:
            x2 = self.extract(x2)
            raw_feats = torch.cat((x1, (x1 + x2) / 2), dim=1)
        else:
            raw_feats = x1

        raw_feats = self.dout(raw_feats)

        ce_fea = self.ce_feat(raw_feats)
        ce_w_fea = self.ce_w_feat(raw_feats)
        mse_feat = self.mse_feat(raw_feats)

        ce_out = self.fc_ce(ce_fea)
        ce_w_out = self.fc_ce_w(ce_w_fea)
        mse_out = self.fc_mse(mse_feat)

        return ce_out, ce_w_out, mse_out
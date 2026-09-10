import sys
import os
import argparse
from Models.CNN_Train import *
from Models.DenseNet import *
from Models.logger import *
from datetime import timedelta
import time

desc = 'FSL_CE_only'

modelName =  'resnet18' # 'shufflenet_v2_x2_0'  #'mobilenet_v2'   
gpuid =   3
pL    =  0.2
useBothEyes = False

dataset =  'DR'        #'LIMUC' # #'

type ='FSL'
PL_TYPE =  'OrdDist' #''AvgProb'  #

if modelName == 'densenet121'  or modelName =='resnet50'  :
    if useBothEyes:
        bs_l = 10
    else:
        if type == 'FSL':
            bs_l = 20
        else:
            bs_l = 8
        bs_u = 12
if modelName == 'resnet18' or modelName =='mobilenet_v2'   :
    if type == 'FSL':
        bs_l = 64
    else:
        bs_l = 16
    bs_u = 3 * bs_l
if modelName == "efficientnet_b0" or modelName =='shufflenet_v2_x2_0':
    if type == 'FSL':
        bs_l = 32
    else:
        bs_l = 8
    bs_u = 3* bs_l
bs_Te = 128
bs_Val = 128

parser = argparse.ArgumentParser()

parser.add_argument("--PL_TYPE", default=PL_TYPE)

parser.add_argument("--dataset", default=dataset)

parser.add_argument("--n_epochs", type=int, default = 70)
parser.add_argument("--fs_epoch", type=int, default =  10)

parser.add_argument("--input_size", type=int, default= 512)
parser.add_argument("--init_thr", type=int, default = 0.30)
parser.add_argument("--minThrClamp", type=int, default = 0.30)
parser.add_argument("--maxThrClamp", type=int, default = 0.95)

parser.add_argument("--desc", default=desc)
parser.add_argument("--bs_l", type=int, default=bs_l)
parser.add_argument("--bs_u", type=int, default=bs_u)
parser.add_argument("--bs_Te", type=int, default=bs_Te)
parser.add_argument("--bs_Val", type=int, default=bs_Val)

parser.add_argument("--modelName", default=modelName)
parser.add_argument("--useBothEyes", default=useBothEyes)
parser.add_argument("--gpuid", type=int, default=gpuid)

parser.add_argument("--lr", type=float, default=5e-3)

parser.add_argument("--pL", default=pL)
 
parser.add_argument("--w_ce",   default= 1)
parser.add_argument("--w_ce_w", default= 0)#5
parser.add_argument("--w_mse",  default= 0)

parser.add_argument("--w_pl",  default=1)

parser.add_argument("--patience", type=int, default= 10)
parser.add_argument("--patience_warmup", type=int, default= 20)

parser.add_argument("--temporal_momentum_ramp_epochs", type=int, default= 30)
parser.add_argument("--temporal_momentum_ramp_iters", type=int, default=None)

parser.add_argument("--momentum_Prob", type=float, default= 0.95)
parser.add_argument("--momentum_Prob_start", type=float, default=0.0)
parser.add_argument("--cw_momentum", type=float, default=   0.90)
parser.add_argument("--momentum_thr", type=float, default=  0.90)

parser.add_argument("--weight_decay", default=5e-4)
parser.add_argument("--type", default=type)
parser.add_argument("--ord_tau", default=0.5)
parser.add_argument("--lamda", default= 0.3)
parser.add_argument("--cw_mode", type=str, default="dynamic", choices=["none", "fixed", "dynamic"])
parser.add_argument("--thr_mode",type=str, default="global_adaptive",choices=["global_adaptive", "classwise_adaptive", "global_fixed"])
parser.add_argument( "--temporal_mode",  type=str, default="both", choices=[ "none", "parameter_ema", "probability_ema", "both"])
parser.add_argument("--prob_source", type=str, default="aggregate", choices=["ce","ce_w","aggregate"], help="Probability source used for temporal EMA: CE, weighted CE, or their aggregation")
opt = parser.parse_args()

dirname = os.path.join('/home/suthakaran/Codes/DR_SSL_V3/Results/VIT/', modelName)
if not os.path.exists(dirname):
    os.makedirs(dirname)
fn = os.path.join(dirname, desc)

print(fn)
sys.stdout = Logger(fn)

def printResults(paraArr, meanArr, stdArr):

    headers = ["Branch", "Acc","BAcc","QWK","MCC","F1μ","F1M", "AUC","Prec","Recall","Sens", "Spec"]
    branches = ["MSE", "CE", "CE_w"]
    widths = [10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    nMetric = 11
    print("\n" + "=" * 175)
    for exp in range(len(paraArr)):
        print(f"\nExperiment {exp+1}")
        print(f"PL = {paraArr[exp][0]}   LR = {paraArr[exp][1]}")
        print("-" * 175)
        header = ""
        for h, w in zip(headers, widths):
            header += f"{h:<{w}}"
        print(header)
        print("-" * 175)
        mean = np.asarray(meanArr[exp])
        std = np.asarray(stdArr[exp])
        for b, branch in enumerate(branches):
            row = f"{branch:<8}"
            start = b * nMetric
            for i in range(start, start + nMetric):
                if np.isnan(mean[i]):
                    txt = "--"
                else:
                    txt = f"{mean[i]:.4f}±{std[i]:.4f}"
                row += f"{txt:<15}"
            print(row)
        print("-" * 175)
    print("=" * 175)

mean_re = []
std_re = []
para = []

save_dir = "/home/suthakaran/Codes/DR_SSL_V3/Saved_models_/"
os.makedirs(save_dir, exist_ok=True)

seeds = [1000, 2000, 3000] #,

for thr_mode in ["global_fixed"]:  
    opt.thr_mode = thr_mode
    for lr in [5e-3]:  
        opt.lr = lr
        seed_results = []
        print("\n" + "=" * 140)
        print(f"Experiment : PL={opt.pL} | LR={opt.lr} | Runs={len(seeds)}")
        print("=" * 140)
        for run, seed in enumerate(seeds, start=1):
            opt.seed = seed
            print(f"\nRun {run}/{len(seeds)} | Seed={seed}")
            print(opt)
            cnn = CNN_Train(opt)
            results, model = cnn.iterate_CNN()
            save_path = os.path.join(save_dir, f"{opt.desc}_seed{seed}.pt")
            torch.save(
                {
                    "seed": seed,
                    "model": model.state_dict(),
                    "results": results,
                },
                save_path,
            )
            seed_results.append(results)
        seed_results = np.asarray(seed_results)
        mean_result = np.mean(seed_results, axis=0)
        std_result = np.std(seed_results, axis=0)
        para.append([opt.pL, opt.lr, opt.type])
        mean_re.append(mean_result)
        std_re.append(std_result)
        print("\n3-Seed Average")
        printResults(para, mean_re, std_re)
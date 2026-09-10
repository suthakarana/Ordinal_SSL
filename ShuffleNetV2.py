import torch
import torch.nn as nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class MLP(nn.Module):
    def __init__(self, dim_in, out_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim_in, out_dim))
    def forward(self, x): return self.layers(x)

class MyShuffleNetV2(nn.Module):
    def __init__(self, num_classes=5, useBothEyes=False, head_dim=512):
        super().__init__()
        self.useBothEyes = useBothEyes
        self.model_name = "shufflenet_v2_x2_0"

        shufflenet = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        nfea = shufflenet.fc.in_features   # 2048
        shufflenet.fc = Identity()
        self.backbone = shufflenet

        self.proj = nn.Sequential(nn.Linear(nfea, head_dim), nn.ReLU(inplace=True))
        times = 2 if useBothEyes else 1
        feature_dim = head_dim * times

        self.dout = nn.Dropout(p=0.2)
        self.mse_feat = MLP(feature_dim, feature_dim, dropout=0.1)
        self.ce_feat = MLP(feature_dim, feature_dim, dropout=0.1)
        self.ce_w_feat = MLP(feature_dim, feature_dim, dropout=0.1)

        self.fc_mse = nn.Linear(feature_dim, 1)
        self.fc_ce = nn.Linear(feature_dim, num_classes)
        self.fc_ce_w = nn.Linear(feature_dim, num_classes)

    def extract(self, x):
        x = self.backbone(x)
        x = self.proj(x)
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
        mse_fea = self.mse_feat(raw_feats)

        ce_out = self.fc_ce(ce_fea)
        ce_w_out = self.fc_ce_w(ce_w_fea)
        mse_out = self.fc_mse(mse_fea)

        return ce_out, ce_w_out, mse_out
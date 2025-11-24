import torch
import torch.nn as nn
import math
from utils import register_model, make_divisible, ChannelShuffle
from blocks import MobileFormerBlock, DynamicActivator, TokenParamGenerator


class MobileFormerHead(nn.Module):
    def __init__(self, in_ch, token_dim, num_classes=1000, groups=1, dropout=0.0):
        super().__init__()

        # Expansion layer (6x) with optional channel shuffle for grouped convolutions
        self.head_exp = nn.Sequential(
            ChannelShuffle(groups) if groups > 1 else nn.Identity(),
            nn.Conv2d(in_ch, in_ch * 6, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(in_ch * 6),
        )

        # Dynamic activation: uses parameters generated from global tokens
        self.head_act = DynamicActivator(mode=2, hidden_scale=2.0)
        self.head_gen = TokenParamGenerator(token_dim, in_ch * 6 * 2)

        final_feat_ch = 1024
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Projection layer to fuse local (image) features and global (token) features
        self.fuse_project = nn.Sequential(
            nn.Linear(in_ch * 6 + token_dim, final_feat_ch),
            nn.BatchNorm1d(final_feat_ch),
            nn.Hardswish(),
        )

        self.cls_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_feat_ch, num_classes)

    def forward(self, inputs):
        x, tokens = inputs

        # 1. Expand local features
        x = self.head_exp(x)

        # 2. Generate activation parameters from tokens and apply to local features
        params = self.head_gen((tokens, None))
        x = self.head_act((x, params))

        # 3. Pool local features
        x = self.avg_pool(x).flatten(1)

        # 4. Extract the first token (classification token)
        cls_token = tokens[0]

        # 5. Concatenate pooled local features with the class token
        fused = torch.cat([x, cls_token], dim=1)

        # 6. Final projection and classification
        fused = self.fuse_project(fused)
        fused = self.cls_dropout(fused)
        return self.fc(fused)


class MobileFormerImpl(nn.Module):
    def __init__(self, config, num_classes=1000, drop_rate=0.0):
        super().__init__()
        width_mult = config["width_mult"]
        token_dim = config["token_dim"]
        num_tokens = config["num_tokens"]
        groups = config.get("groups", 1)

        if "stem_ch" in config:
            stem_ch = config["stem_ch"]
        else:
            stem_ch = make_divisible(16 * width_mult)

        # Standard convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU6(inplace=True),
        )

        # Learnable global tokens (Conceptually similar to ViT's CLS token but multiple)
        self.tokens = nn.Embedding(num_tokens, token_dim)

        layers = []
        in_ch = stem_ch

        # Hardcoded resolution map for Mobile-Former bridge calculations
        resolutions = [
            56 * 56,
            28 * 28,
            28 * 28,
            14 * 14,
            14 * 14,
            14 * 14,
            14 * 14,
            7 * 7,
            7 * 7,
            7 * 7,
            7 * 7,
        ]

        while len(resolutions) < len(config["layers"]):
            resolutions.append(7 * 7)

        # Build the parallel Mobile-Former blocks
        for i, cfg in enumerate(config["layers"]):
            out_ch = make_divisible(cfg["out"] * width_mult)
            stride = cfg["stride"]
            expand = cfg["exp"]
            current_res = resolutions[i]

            block_args = {
                "in_ch": in_ch,
                "out_ch": out_ch,
                "stride": stride,
                "expand_ratio": expand,
                "token_dim": token_dim,
                "num_tokens": num_tokens,
                "resolution": current_res,
                "se_cfg": [2, 0, 2, 0],  # Specific config for SE block placement
                "k_size": (3, 3),
                "groups": groups,
                "drop": drop_rate,
            }

            layers.append(MobileFormerBlock(**block_args))
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        self.classifier = MobileFormerHead(
            in_ch=in_ch,
            token_dim=token_dim,
            num_classes=num_classes,
            groups=groups,
            dropout=drop_rate,
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B = x.shape[0]

        # Expand learnable tokens to batch size: (Num_Tokens, Dim) -> (B, Num_Tokens, Dim)
        # Note: Permute handles the internal dimension logic of the blocks
        tokens = self.tokens.weight.unsqueeze(0).repeat(B, 1, 1).permute(1, 0, 2)

        x = self.stem(x)

        # Pass both feature map (x) and global tokens through the parallel backbone
        # The blocks facilitate bidirectional info flow: Mobile -> Former and Former -> Mobile
        x, tokens = self.backbone((x, tokens))

        return self.classifier((x, tokens))


@register_model
def mobile_former_26m(**kwargs):
    # Configuration for the 26M FLOPs variant
    config = {
        "width_mult": 1.0,
        "token_dim": 192,
        "num_tokens": 6,
        "stem_ch": 16,
        "groups": 4,
        "layers": [
            {"exp": 2, "out": 16, "stride": 1},
            {"exp": 2, "out": 24, "stride": 2},
            {"exp": 2, "out": 24, "stride": 1},
            {"exp": 2, "out": 48, "stride": 2},
            {"exp": 2, "out": 48, "stride": 1},
            {"exp": 3, "out": 96, "stride": 2},
            {"exp": 3, "out": 96, "stride": 1},
            {"exp": 3, "out": 96, "stride": 1},
            {"exp": 3, "out": 96, "stride": 1},
            {"exp": 3, "out": 128, "stride": 2},
            {"exp": 3, "out": 128, "stride": 1},
        ],
    }

    drop_rate = kwargs.pop("drop_rate", 0.0)

    return MobileFormerImpl(
        config, num_classes=kwargs.get("num_classes", 1000), drop_rate=drop_rate
    )

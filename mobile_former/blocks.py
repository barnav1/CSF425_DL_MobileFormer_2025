import torch
import torch.nn as nn
from utils import DropPath, ChannelShuffle


class DynamicActivator(nn.Module):
    """
    Handles Dynamic ReLU (DyReLU) or Squeeze-and-Excitation (SE) scaling.
    """

    def __init__(self, mode=2, hidden_scale=2.0, is_se=False):
        super().__init__()
        self.mode = mode
        self.scale = hidden_scale
        self.base_act = nn.ReLU6(inplace=True)
        self.use_base = (mode == 0) or (is_se and mode == 1)

    def forward(self, x):
        # Unpack features and coefficients (generated from tokens)
        if isinstance(x, tuple):
            feat, coeffs = x
        else:
            feat, coeffs = x, None

        if self.use_base:
            feat = self.base_act(feat)

        # Mode 1: Simple channel scaling (SE-like)
        if self.mode == 1:
            return feat * (coeffs * self.scale)

        # Mode 2: Dynamic ReLU -> max(a*x + b, c*x + d)
        elif self.mode == 2:
            c_dim = coeffs.shape[1] // 2
            alpha, beta = torch.split(coeffs, [c_dim, c_dim], dim=1)

            # Scale coefficients to range centered around initialization
            a_scaled = (alpha - 0.5) * self.scale + 1.0
            b_scaled = (beta - 0.5) * self.scale
            return torch.max(feat * a_scaled, feat * b_scaled)
        return feat


class TokenParamGenerator(nn.Module):
    """
    Maps a global token to dynamic parameters (weights/biases) for the local feature map.
    """

    def __init__(self, in_dim, out_dim, tok_idx=0, reduction=4):
        super().__init__()
        self.idx = tok_idx
        mid_dim = in_dim // reduction
        self.map = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
            nn.Hardsigmoid(),
        )

    def forward(self, inputs):
        tokens, _ = inputs
        # Select specific token responsible for this layer
        selected = tokens[self.idx]
        params = self.map(selected)
        # Reshape for broadcasting over spatial dimensions (B, C, 1, 1)
        return params[:, :, None, None]


class LocalToGlobalInterface(nn.Module):
    """
    Bridge: Updates global tokens using local feature map context (Mobile -> Former).
    """

    def __init__(self, dim, token_dim, num_tokens, heads=2, drop=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(token_dim, dim)  # Query comes from Tokens
        self.to_out = nn.Linear(dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = DropPath(drop)

    def forward(self, x):
        feature_map, tokens = x
        B, C, H, W = feature_map.shape
        T = tokens.shape[0]

        # Q: Global Tokens
        q = self.to_q(tokens).view(T, B, self.heads, -1).permute(1, 2, 0, 3)
        # K: Local Feature Map (flattened spatial dims)
        k = feature_map.view(B, self.heads, -1, H * W)

        # Cross-Attention: Tokens attend to Feature Map
        attn_scores = (q @ k) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        # V: Local Feature Map (transposed for matrix mult)
        context = attn_probs @ k.transpose(-1, -2)
        context = context.permute(2, 0, 1, 3).reshape(T, B, -1)

        out = self.to_out(context)
        tokens = self.norm(tokens + self.dropout(out))

        return tokens, attn_scores


class GlobalToLocalInterface(nn.Module):
    """
    Bridge: Updates local feature map using global token context (Former -> Mobile).
    """

    def __init__(self, dim, resolution, token_dim, num_tokens, heads=2, drop=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.num_tokens = num_tokens
        # K, V come from Tokens
        self.to_k = nn.Linear(token_dim, dim)
        self.to_v = nn.Linear(token_dim, dim)
        self.dropout = DropPath(drop)

    def forward(self, x):
        feature_map, tokens = x
        B, C, H, W = feature_map.shape

        v = self.to_v(tokens).permute(1, 2, 0)
        k = self.to_k(tokens).permute(1, 2, 0).view(B, self.heads, -1, self.num_tokens)
        # Q: Local Feature Map
        q = feature_map.view(B, self.heads, -1, H * W).transpose(-1, -2)

        # Cross-Attention: Feature Map attends to Tokens
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        v_reshaped = v.view(B, self.heads, -1, self.num_tokens)
        out = attn @ v_reshaped.transpose(-1, -2)
        spatial_out = out.transpose(-1, -2).reshape(B, C, H, W)

        return feature_map + self.dropout(spatial_out)


class GlobalProcessor(nn.Module):
    """
    Standard Transformer Self-Attention block for processing global tokens.
    """

    def __init__(self, token_dim, num_tokens, heads=4, drop=0.0):
        super().__init__()
        self.scale = (token_dim // heads) ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(token_dim, token_dim)
        self.to_out = nn.Linear(token_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.drop = DropPath(drop)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.ffn_norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        T, B, C = x.shape
        # Standard Multi-Head Self-Attention (MHSA)
        q = self.to_q(x).view(T, B, self.heads, -1).permute(1, 2, 0, 3)
        k = x.permute(1, 2, 0).view(B, self.heads, -1, T)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ k.transpose(-1, -2)).permute(2, 0, 1, 3).reshape(T, B, -1)

        x = self.norm(x + self.drop(self.to_out(out)))
        x = self.ffn_norm(x + self.ffn(x))
        return x


class MobileFormerBlock(nn.Module):
    """
    Parallel block: Computes MobileNet (Local) and Transformer (Global) branches,
    exchanging info via bridges.
    """

    def __init__(
        self, in_ch, out_ch, stride, expand_ratio, k_size, se_cfg, groups=1, **kwargs
    ):
        super().__init__()
        self.use_skip = stride == 1 and in_ch == out_ch
        mid_ch = int(in_ch * expand_ratio)
        kernel_s = k_size[0] if isinstance(k_size, tuple) else k_size
        padding = kernel_s // 2

        # 1. Bridge: Local features -> Global tokens
        self.bridge_in = LocalToGlobalInterface(
            in_ch,
            kwargs["token_dim"],
            kwargs["num_tokens"],
            drop=kwargs.get("drop", 0.0),
        )

        # 2. Global Processor (Transformer on tokens)
        self.token_mixer = GlobalProcessor(
            kwargs["token_dim"], kwargs["num_tokens"], drop=kwargs.get("drop", 0.0)
        )

        # 3. Bridge: Global tokens -> Local features
        out_res = kwargs["resolution"] // (stride * stride)
        self.bridge_out = GlobalToLocalInterface(
            out_ch,
            out_res,
            kwargs["token_dim"],
            kwargs["num_tokens"],
            drop=kwargs.get("drop", 0.0),
        )

        # MobileNetV2-style Inverted Residual Layers
        self.pw_exp = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(mid_ch),
            ChannelShuffle(groups) if groups > 1 else nn.Identity(),
        )

        # Dynamic Activator 1 (Pointwise Expansion)
        self.act1 = DynamicActivator(mode=se_cfg[0], is_se=True)
        self.gen1 = (
            TokenParamGenerator(kwargs["token_dim"], mid_ch * se_cfg[0])
            if se_cfg[0] > 0
            else None
        )

        self.dw = nn.Sequential(
            nn.Conv2d(
                mid_ch, mid_ch, kernel_s, stride, padding, groups=mid_ch, bias=False
            ),
            nn.BatchNorm2d(mid_ch),
        )

        # Dynamic Activator 2 (Depthwise)
        self.act2 = DynamicActivator(mode=se_cfg[2], is_se=True)
        self.gen2 = (
            TokenParamGenerator(kwargs["token_dim"], mid_ch * se_cfg[2])
            if se_cfg[2] > 0
            else None
        )

        self.pw_proj = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            ChannelShuffle(groups) if groups > 1 else nn.Identity(),
        )

        # Dynamic Activator 3 (Pointwise Projection)
        act_mode = 1 if se_cfg[3] == 1 else -1
        self.act3 = DynamicActivator(mode=act_mode)
        self.gen3 = (
            TokenParamGenerator(kwargs["token_dim"], out_ch * act_mode)
            if se_cfg[3] > 0
            else None
        )

    def forward(self, x):
        feat, tokens = x

        # Step 1: Update tokens using input feature map
        tokens, attn_map = self.bridge_in((feat, tokens))

        # Step 2: Process tokens (Transformer part)
        tokens = self.token_mixer(tokens)

        # Step 3: MobileNet Conv layers (gated by token-generated params)
        out = self.pw_exp(feat)
        if self.gen1:
            out = self.act1((out, self.gen1((tokens, attn_map))))
        else:
            out = self.act1(out)

        out = self.dw(out)
        if self.gen2:
            out = self.act2((out, self.gen2((tokens, attn_map))))
        else:
            out = self.act2(out)

        out = self.pw_proj(out)
        if self.gen3:
            out = self.act3((out, self.gen3((tokens, attn_map))))
        else:
            out = self.act3(out)

        # Step 4: Update feature map using processed tokens
        out = self.bridge_out((out, tokens))

        if self.use_skip:
            out = out + feat

        return (out, tokens)

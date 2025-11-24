import math
import torch
import torch.nn as nn


def register_model(fn):
    # Placeholder decorator for model registry
    return fn


def make_divisible(v, divisor=8, min_value=None):
    """Ensures 'v' is divisible by 'divisor' (common for efficient hardware execution)."""
    if min_value is None:
        min_value = divisor
    # Round to the nearest divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Ensure rounding down doesn't decrease value by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DropPath(nn.Module):
    """Implements Stochastic Depth (dropping entire residual paths) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Create shape (N, 1, 1, 1) to apply mask per sample, not per element
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize to 0 or 1
        # Apply mask and scale to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


class ChannelShuffle(nn.Module):
    """Interleaves channels to allow information exchange between groups (ShuffleNet)."""

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        if self.groups == 1:
            return x
        b, c, h, w = x.shape
        c_per_group = c // self.groups

        # Reshape to: (Batch, Groups, Channels_per_Group, Height, Width)
        x = x.view(b, self.groups, c_per_group, h, w)

        # Transpose groups and channels_per_group to mix them
        x = x.transpose(1, 2).contiguous()

        # Flatten back to original shape
        return x.view(b, -1, h, w)

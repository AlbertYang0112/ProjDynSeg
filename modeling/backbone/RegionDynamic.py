import torch as th
from math import ceil
from torch import nn
from torch.nn.functional import pad, fold, unfold
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec


class DynamicBlock(nn.Module):
    def __init__(self, in_channels, shared_block, extra_block,
                 region_cnt_w, region_cnt_h) -> None:
        super().__init__()
        self.shared_block = shared_block
        self.extra_block = extra_block
        self.switch_module = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(region_cnt_h, region_cnt_w)),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, groups=in_channels // 2),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.ReLU6(inplace=True),
        )
        self.region_cnt_w = region_cnt_w
        self.region_cnt_h = region_cnt_h

    def region_unfold(self, x):
        """
        Crop the input feature into (region_cnt_w, region_cnt_h) blocks
        """
        _, c, h, w = x.shape
        assert h % self.region_cnt_h == 0 and w % self.region_cnt_w == 0
        region_h = h // self.region_cnt_h
        region_w = w // self.region_cnt_w
        unfold_x = unfold(x, kernel_size=(region_h, region_w), stride=(region_h, region_w))
        unfold_x = unfold_x.transpose(-1, -2)
        unfold_x = unfold_x.reshape(-1, c, region_h, region_w)
        return unfold_x, (region_h, region_w)

    def region_padding(self, x):
        h, w = x.shape[-2:]
        pad_h = ceil(h / self.region_cnt_h) * self.region_cnt_h - h
        pad_w = ceil(w / self.region_cnt_w) * self.region_cnt_w - w
        pad_factor = (pad_h // 2, pad_h - (pad_h // 2), pad_w, pad_w - (pad_w // 2))
        x_pad = pad(x, pad_factor, mode='replicate')
        return x_pad, pad_factor

    def region_stripping(self, x, pad_factor):
        h_start, h_end, w_start, w_end = pad_factor
        h, w = x.shape[-2:]
        return x[..., h_start:h-h_end, w_start:w-w_end]

    def forward(self, x):
        batch_size = x.shape[0]
        x_pad, pad_factor = self.region_padding(x)
        pad_shape = x_pad.shape[-2:]
        patches, region_shape = self.region_unfold(x_pad)
        shared_feat = self.shared_block(x_pad)
        patch_feat = self.extra_block(patches)
        patch_feat = patch_feat.reshape(batch_size, self.region_cnt_h*self.region_cnt_w, -1)
        patch_feat = patch_feat.transpose(-1, -2)
        patch_feat = fold(patch_feat, output_size=pad_shape,
                          kernel_size=region_shape, stride=region_shape)
        patch_channel = patch_feat.shape[1]
        switch = self.switch_module(x_pad)
        switch = th.cat([switch] * patch_channel * region_shape[0] * region_shape[1], 
                        dim=1).reshape(
                            batch_size, -1, self.region_cnt_w * self.region_cnt_h)
        print(switch.shape)
        switch = fold(switch, output_size=pad_shape, 
                      kernel_size=region_shape, stride=region_shape)
        print(patch_channel, switch.shape, patch_feat.shape, shared_feat.shape)
        out_feat = (1 - switch) * shared_feat + switch * patch_feat
        out_feat = self.region_stripping(out_feat, pad_factor)
        return out_feat

import torch as th
from math import ceil
from torch import nn
from torch.nn.functional import pad, fold, unfold
from detectron2.modeling.backbone.resnet import build_resnet_backbone, BottleneckBlock
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
import pdb


class DynamicBlock(nn.Module):
    def __init__(self, in_channels, shared_block, extra_block, stride,
                 region_cnt_w, region_cnt_h) -> None:
        super().__init__()
        self.shared_block = shared_block
        self.extra_block = extra_block
        self.stride = stride
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
        self.stride_layer = nn.MaxPool2d(kernel_size=3, stride=self.stride, padding=1)
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
        pad_factor = (pad_h // 2, pad_h - (pad_h // 2), pad_w // 2, pad_w - (pad_w // 2))
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
        if self.stride != 1:
            patch_feat = self.stride_layer(patch_feat)
        patch_channel = patch_feat.shape[1]
        switch = self.switch_module(x_pad)
        switch_expand = th.cat([switch] * patch_channel * region_shape[0] * region_shape[1],
                               dim=1).reshape(
                               batch_size, -1, self.region_cnt_w * self.region_cnt_h)
        switch_expand = fold(switch_expand, output_size=pad_shape, 
                             kernel_size=region_shape, stride=region_shape)
        if self.stride != 1:
            switch_expand = self.stride_layer(switch_expand)
        print(switch_expand.shape, shared_feat.shape, patch_feat.shape)
        out_feat = (1 - switch_expand) * shared_feat + switch_expand * patch_feat
        out_feat = self.region_stripping(out_feat, pad_factor)
        return out_feat, switch


class DynamicResNet(nn.Module):
    def __init__(self, resnet) -> None:
        super().__init__()
        self.stem = resnet.stem
        for i in range(2, 5):
            res_layer = getattr(resnet, f'res{i}')
            in_channels = res_layer[0].in_channels
            out_channels = res_layer[0].out_channels
            stride = res_layer[0].stride
            bottleneck_channels = in_channels
            extra_layer = BottleneckBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          bottleneck_channels=bottleneck_channels)
            dynamic_layer = DynamicBlock(in_channels=in_channels,
                                         shared_block=res_layer, extra_block=extra_layer,
                                         stride=stride,
                                         region_cnt_w=8, region_cnt_h=8)
            setattr(self, f"dynamic_layer{i}", dynamic_layer)

    def forward(self, x):
        x = self.stem(x)
        outputs = {}
        for i in range(2, 5):
            layer = getattr(self, f"dynamic_layer{i}")
            print(f"Layer {i}")
            x, sw = layer(x)
            outputs[f'res{i}'] = x
            outputs[f'sw{i}'] = sw
        return outputs


@BACKBONE_REGISTRY.register()
def build_dynamic_resnet_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must
        be a subclass of :class:`Backbone`.
    """
    resnet = build_resnet_backbone(cfg, input_shape)
    backbone = DynamicResNet(resnet)
    return backbone

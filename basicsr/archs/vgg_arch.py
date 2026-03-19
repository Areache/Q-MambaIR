import os
import torch
from torch import nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """VGG network for perceptual feature extraction.

    This implementation is compatible with BasicSR's ``PerceptualLoss``:
    it accepts layer names like ``'conv1_2'``, ``'conv2_2'``, ``'conv3_4'``,
    ``'conv4_4'``, ``'conv5_4'`` and returns a dict {layer_name: feature}.

    Args:
        layer_name_list (list[str]): e.g. ['conv1_2', 'conv2_2', 'conv3_4', ...].
        vgg_type (str): 'vgg19' or 'vgg16'. Default: 'vgg19'.
        use_input_norm (bool): If True, apply ImageNet mean/std normalization.
        range_norm (bool): If True, map [-1, 1] -> [0, 1] before normalization.
        weight_path (str, optional): Path to manually downloaded pretrained weights.
            If provided, will load from this path instead of downloading.
    """

    def __init__(
        self,
        layer_name_list,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        weight_path: str = None,
    ):
        super().__init__()

        # ---- build backbone ----
        if vgg_type == "vgg19":
            vgg = models.vgg19(pretrained=False)  # 先创建模型，不加载权重
        elif vgg_type == "vgg16":
            vgg = models.vgg16(pretrained=False)
        else:
            raise ValueError(f"Unsupported vgg_type: {vgg_type}")

        # ---- load pretrained weights ----
        # 优先级: 手动权重路径 > 自动下载 > 无预训练权重
        weights_loaded = False
        
        if weight_path is not None and os.path.isfile(weight_path):
            try:
                print(f"[VGGFeatureExtractor] Loading weights from: {weight_path}")
                state_dict = torch.load(weight_path, map_location='cpu')
                vgg.load_state_dict(state_dict)
                weights_loaded = True
                print(f"[VGGFeatureExtractor] ✓ Successfully loaded manual weights")
            except Exception as e:
                print(f"[VGGFeatureExtractor] ✗ Failed to load manual weights: {e}")
                print(f"[VGGFeatureExtractor] Falling back to automatic download...")
        
        if not weights_loaded:
            try:
                # 尝试自动下载预训练权重
                if vgg_type == "vgg19":
                    vgg = models.vgg19(pretrained=True)
                elif vgg_type == "vgg16":
                    vgg = models.vgg16(pretrained=True)
                weights_loaded = True
                print(f"[VGGFeatureExtractor] ✓ Successfully loaded pretrained weights (auto-download)")
            except Exception as e:
                print(f"[VGGFeatureExtractor] ⚠ Could not load pretrained weights: {e}")
                print(f"[VGGFeatureExtractor] Using randomly initialized weights (performance may be degraded)")

        self.features = vgg.features
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        # ---- input normalization config ----
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        if self.use_input_norm:
            # ImageNet statistics
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        # ---- map layer names to indices in vgg.features ----
        self.layer_name_list = layer_name_list
        # This mapping follows the standard VGG19 layout used in BasicSR
        self.layer2idx = {
            "conv1_1": 0,
            "conv1_2": 2,
            "conv2_1": 5,
            "conv2_2": 7,
            "conv3_1": 10,
            "conv3_2": 12,
            "conv3_3": 14,
            "conv3_4": 16,
            "conv4_1": 19,
            "conv4_2": 21,
            "conv4_3": 23,
            "conv4_4": 25,
            "conv5_1": 28,
            "conv5_2": 30,
            "conv5_3": 32,
            "conv5_4": 34,
        }

        # 只跑到所需的最大层，避免多余计算
        try:
            self.max_idx = max(self.layer2idx[name] for name in self.layer_name_list)
        except KeyError as e:
            raise KeyError(f"Unsupported VGG layer name in layer_name_list: {e}")

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
            x (Tensor): (N, 3, H, W), RGB, [0,1] 或 [-1,1] 取决于 range_norm。

        Returns:
            dict[str, Tensor]: {layer_name: feature}
        """
        if self.range_norm:
            # [-1, 1] -> [0, 1]
            x = (x + 1.0) / 2.0

        if self.use_input_norm:
            x = (x - self.mean) / self.std

        out = {}
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            for name in self.layer_name_list:
                if i == self.layer2idx.get(name, -1):
                    out[name] = h
            if i >= self.max_idx:
                break

        return out



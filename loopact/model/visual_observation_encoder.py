import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualObservationEncoder(nn.Module):
    """
    ResNet18-based VisualObservationEncoder that returns features with shape (B, 512, 8, 8).

    Input:
      obs: (B, 3, 224, 224)  float in [0,1] or [0,255]

    Output:
      feat: (B, 512, 8, 8)

    Notes:
    - Uses torchvision ResNet18 pretrained on ImageNet.
    - Outputs 512 channels (vs 768 for DINOv2).
    - Outputs 8x8 spatial grid (vs 7x7 for DINOv2).
    - Much faster than DINOv2 (~2-3ms vs ~15-17ms).
    - Freezes backbone by default (fast + stable).
    - ALWAYS reloads ImageNet weights on first forward to avoid checkpoint corruption.
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model_name: str = "resnet18", freeze: bool = True):
        super().__init__()
        from torchvision import models

        self.model_name = model_name
        self.freeze = freeze
        self._weights_loaded = False

        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove avgpool and fc layers - we want spatial features
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Output channels: 512 (from layer4)
        # Output spatial: 7x7 for 224x224 input, we'll pool to 8x8

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

        print(f"✓ VisualObservationEncoder initialized: ResNet18 (freeze={freeze})")

    def _ensure_weights_loaded(self, device):
        """
        ALWAYS reload ResNet18 pretrained weights on first forward pass.

        Why? HuggingFace's from_pretrained can corrupt these weights by:
        1. Using meta tensors (lazy loading)
        2. Overwriting with garbage from checkpoint (if keys match)

        Since ResNet18 is frozen and should always use ImageNet weights,
        we reload them unconditionally on first forward to guarantee correctness.
        """
        if self._weights_loaded:
            return

        # Always reload fresh ImageNet weights on first forward
        print("⚠ Reloading ResNet18 ImageNet pretrained weights...")
        self._reload_weights(device)
        self._weights_loaded = True

    def _reload_weights(self, device):
        """Reload fresh ResNet18 pretrained weights."""
        from torchvision import models

        # Load fresh pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Copy weights to current modules
        self.conv1.load_state_dict(resnet.conv1.state_dict())
        self.bn1.load_state_dict(resnet.bn1.state_dict())
        self.layer1.load_state_dict(resnet.layer1.state_dict())
        self.layer2.load_state_dict(resnet.layer2.state_dict())
        self.layer3.load_state_dict(resnet.layer3.state_dict())
        self.layer4.load_state_dict(resnet.layer4.state_dict())

        # Move to correct device
        self.to(device)

        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

        # Verify weights are valid
        sample_param = self.conv1.weight
        if torch.isnan(sample_param).any():
            raise RuntimeError("ResNet18 weights still NaN after reload!")

        print(f"✓ ResNet18 ImageNet weights loaded on {device}")
        print(
            f"  conv1.weight range: [{sample_param.min().item():.3f}, {sample_param.max().item():.3f}]"
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            x = x.float()
        # If input looks like [0,255], rescale to [0,1]
        if x.max() > 1.5:
            x = x / 255.0
        mean = self.IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        std = self.IMAGENET_STD.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, 3, 224, 224)
        returns: (B, 512, 8, 8)
        """
        # Always reload ImageNet weights on first forward (checkpoint may have corrupted them)
        self._ensure_weights_loaded(obs.device)

        x = self._normalize(obs)

        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)  # (B, 512, 7, 7) for 224x224 input
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        # Pool to 8x8 for finer spatial resolution
        features = F.adaptive_avg_pool2d(x, (8, 8))  # (B, 512, 8, 8)

        return features


# Serialization-compatible class aliases.
ObsEncoder = VisualObservationEncoder

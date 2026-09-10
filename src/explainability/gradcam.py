"""Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet."""

from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM visual explainer for Convolutional Feature Extractors.

    Supports both standard classification heads and CORAL ordinal regression heads.
    For CORAL, can target a specific ordinal threshold logit (e.g., grade >= k).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        """Initialize GradCAM hook manager.

        Args:
            model: PyTorch DRSeverityModel.
            target_layer: Specific layer to hook. If None, automatically finds
                          the last Conv2d layer in the backbone.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Find last Conv2d in the model backbone
            target_layer = self._find_last_conv_layer(model)

        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Locate the deepest Conv2d layer in the module hierarchy."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in the provided model.")
        return last_conv

    def _forward_hook(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_index: Optional[int] = None,
        is_ordinal: bool = False,
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap.

        Args:
            image_tensor: Input tensor of shape (1, C, H, W).
            target_index: Target class index (for CE) or threshold index (for CORAL).
                          If None, uses the predicted grade.
            is_ordinal: If True, treats logits as CORAL threshold logits.

        Returns:
            Normalized 2D numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(image_tensor)

        if is_ordinal:
            if target_index is None:
                # Target the highest active threshold or sum of active sigmoids
                probs = torch.sigmoid(logits)
                target_score = logits.sum()
            else:
                # Target specific threshold logit (e.g. grade >= target_index)
                idx = min(target_index, logits.size(1) - 1)
                target_score = logits[0, idx]
        else:
            if target_index is None:
                target_index = logits.argmax(dim=1).item()
            target_score = logits[0, target_index]

        # Backward pass to get gradients at target layer
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations in GradCAM hooks.")

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_f, W_f)

        # Apply ReLU to retain only positive contributions
        cam = torch.clamp(cam, min=0.0)

        # Convert to numpy and resize to original input resolution
        cam_np = cam.squeeze().cpu().numpy()
        h_orig, w_orig = image_tensor.shape[2], image_tensor.shape[3]

        if np.max(cam_np) > 0:
            cam_np = cam_np / np.max(cam_np)

        resized_cam = cv2.resize(cam_np, (w_orig, h_orig))
        return resized_cam

    def remove_hooks(self) -> None:
        """Clean up hook handles."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_gradcam(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend Grad-CAM heatmap over base retinal image.

    Args:
        base_image: RGB uint8 numpy array of shape (H, W, 3).
        heatmap: Float numpy array of shape (H, W) with values in [0, 1].
        alpha: Blending weight for heatmap.
        colormap: OpenCV colormap enum.

    Returns:
        Blended RGB uint8 numpy array of shape (H, W, 3).
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    if base_image.shape[:2] != heatmap.shape[:2]:
        base_image = cv2.resize(base_image, (heatmap.shape[1], heatmap.shape[0]))

    overlay = cv2.addWeighted(base_image, 1.0 - alpha, colored_heatmap, alpha, 0)
    return overlay

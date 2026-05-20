"""
Shared Grad-CAM utility for all 12 CNN architectures.
Based on: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Uses a tensor-level gradient hook (registered inside the forward hook on
    the activation tensor itself) instead of register_full_backward_hook.
    This avoids a RuntimeError that occurs when the next module after the
    target layer uses ReLU(inplace=True) – as AlexNet, VGG, and MobileNetV2
    all do – which would conflict with the BackwardHookFunctionBackward
    wrapper that register_full_backward_hook inserts.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._grad_handle = None   # tensor-level hook handle (set per-call)
        # Store module-level forward hook handle so it can be removed cleanly.
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        # NOTE: No register_full_backward_hook here – gradients are captured
        # via a tensor hook registered inside _fwd_hook instead.

    def remove_hooks(self):
        """Remove all registered hooks. Call after inference to avoid leaks."""
        self._fwd_handle.remove()
        if self._grad_handle is not None:
            self._grad_handle.remove()
            self._grad_handle = None

    def _fwd_hook(self, module, inp, out):
        # Clone so that any subsequent inplace op (e.g. ReLU_) on `out` does
        # not corrupt the captured activation tensor.
        self.activations = out.clone()
        # Register a tensor-level gradient hook directly on the (non-cloned)
        # live activation. This captures d(loss)/d(activation) during backward
        # without inserting an extra autograd node that would conflict with
        # inplace operations downstream.
        if self._grad_handle is not None:
            self._grad_handle.remove()
        self._grad_handle = out.register_hook(
            lambda grad: setattr(self, 'gradients', grad.detach().clone())
        )

    def generate(self, input_tensor, target_class=None):
        """
        Returns:
            cam (np.ndarray): normalised heatmap [H, W] in [0,1]
            predicted_class (int)
            confidence (float)
            probabilities (np.ndarray): softmax scores
        """
        self.model.eval()
        # Ensure gradients flow through the input so the tensor hook fires.
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        # inception_v3 returns (out, aux) in train mode – handle both
        if isinstance(output, tuple):
            output = output[0]

        probs = F.softmax(output[0], dim=0)
        pred_cls = int(torch.argmax(probs))
        confidence = float(probs[pred_cls])

        if target_class is None:
            target_class = pred_cls

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations.detach(), dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), pred_cls, confidence, probs.detach().cpu().numpy()


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray,
                alpha: float = 0.5) -> tuple:
    """
    Overlay CAM heatmap on image.
    image_rgb: H×W×3 float32 in [0,1]
    cam: H×W float32 in [0,1]
    Returns (overlay, heatmap, cam_resized) all as float32 [0,1] arrays.
    """
    h, w = image_rgb.shape[:2]
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = np.array(cam_pil.resize((w, h), Image.BILINEAR)) / 255.0

    cmap = plt.colormaps.get_cmap('jet')
    heatmap = cmap(cam_resized)[:, :, :3]

    if image_rgb.max() > 1:
        image_rgb = image_rgb / 255.0

    overlay = np.clip((1 - alpha) * image_rgb + alpha * heatmap, 0, 1)
    return overlay, heatmap, cam_resized


def generate_gradcam_figure(original_image: Image.Image, cam: np.ndarray,
                            size: int = 224) -> plt.Figure:
    """Return a matplotlib Figure with 3-panel Grad-CAM visualisation."""
    img_arr = np.array(original_image.resize((size, size))) / 255.0
    overlay, heatmap, cam_resized = overlay_cam(img_arr, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_arr);       axes[0].set_title('Original');      axes[0].axis('off')
    axes[1].imshow(cam_resized, cmap='jet'); axes[1].set_title('Grad-CAM Heatmap'); axes[1].axis('off')
    axes[2].imshow(overlay);       axes[2].set_title('Overlay');       axes[2].axis('off')
    plt.tight_layout()
    return fig

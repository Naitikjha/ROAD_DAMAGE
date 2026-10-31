"""
Explainable AI (XAI): Grad-CAM Implementation for ResPotholeNet.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM):
Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Hooks into the final convolutional feature maps of ResPotholeNet (SPPF layer)
to visualize what visual features (cavity borders, shadows, asphalt texture)
the CNN attends to when computing pothole confidence.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM explanation generator for ResPotholeNet.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()

        # If no target layer specified, use the final conv layer of SPPF
        if target_layer is None:
            if hasattr(model, "sppf") and hasattr(model.sppf, "cv2"):
                self.target_layer = model.sppf.cv2.conv
            elif hasattr(model, "stage3"):
                self.target_layer = model.stage3[-1].conv2.conv
            else:
                raise ValueError("Target layer could not be automatically identified.")
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a normalized Grad-CAM heatmap for a single input patch.

        Parameters
        ----------
        input_tensor : torch.Tensor of shape (1, 3, H, W)

        Returns
        -------
        heatmap : np.ndarray of shape (H, W) in [0, 1]
        """
        self.model.zero_grad()
        conf, _ = self.model(input_tensor)

        # Target score for pothole probability
        score = conf[0, 0]
        score.backward(retain_graph=True)

        # Global average pool the gradients
        # gradients shape: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU to retain only features that have positive influence
        cam = F.relu(cam)

        # Interpolate to input resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        patch_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap onto the original BGR patch.

        Parameters
        ----------
        patch_bgr : np.ndarray (H, W, 3) in uint8 [0, 255]
        heatmap   : np.ndarray (H, W) in float [0, 1]
        alpha     : blending ratio (0 = only original image, 1 = only heatmap)

        Returns
        -------
        blended : np.ndarray (H, W, 3) in uint8 [0, 255]
        """
        h, w = patch_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        blended = cv2.addWeighted(color_heatmap, alpha, patch_bgr, 1.0 - alpha, 0)
        return blended

    def generate_cam(
        self,
        crop_bgr: np.ndarray,
        alpha: float = 0.5,
    ) -> tuple:
        """
        Generate Grad-CAM visualization from a raw BGR image patch.

        Parameters
        ----------
        crop_bgr : np.ndarray
            BGR image patch from candidate detection.
        alpha : float
            Alpha blending value for heatmap overlay.

        Returns
        -------
        tuple of (heatmap_rgb, overlay_bgr, confidence_score)
        """
        from detector import normalize_patch

        h, w = crop_bgr.shape[:2]
        resized = cv2.resize(crop_bgr, (128, 128))
        norm = normalize_patch(resized)

        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        heatmap = self.generate_heatmap(tensor)
        heatmap_resized = cv2.resize(heatmap, (w, h))
        overlay = self.overlay_heatmap(crop_bgr, heatmap_resized, alpha=alpha)

        # Heatmap RGB for direct streamlit display
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_rgb = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            conf, _ = self.model(tensor)
            score = float(conf[0, 0].cpu().item())

        return heatmap_rgb, overlay, score

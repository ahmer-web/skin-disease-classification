import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output

        # Hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        output[0, pred_class].backward()

        gradients = self.gradients
        activations = self.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().numpy(), 0)

        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        return heatmap


# ✅ Helper function to generate heatmap + overlay
def generate_gradcam(image_tensor, model, target_layer):
    """Generate a Grad-CAM heatmap and overlay image."""
    grad_cam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap = grad_cam.generate(image_tensor)

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert tensor back to image
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
    image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    image_np = np.clip(image_np, 0, 1)
    image_np = np.uint8(image_np * 255)

    # Overlay heatmap onto original image
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)

    # Convert to PIL for Streamlit
    heatmap_img = Image.fromarray(heatmap_colored)
    overlay_img = Image.fromarray(overlay)

    return heatmap_img, overlay_img
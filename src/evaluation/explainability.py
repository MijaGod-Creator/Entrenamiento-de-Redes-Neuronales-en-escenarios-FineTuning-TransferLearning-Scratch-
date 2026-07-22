from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.config.experiment import ExperimentConfig
from src.config.settings import EXPLAINABILITY_FIGURES_DIR, RAFDB_LABEL_MAP
from src.models.model_factory import build_model


def _load_checkpoint(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    config = ExperimentConfig(**checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def load_image_for_model(path: str, image_size: tuple[int, int]) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image)


def find_last_conv_layer(model: torch.nn.Module):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_module = module
    if last_name is None or last_module is None:
        raise ValueError("No se encontro una capa Conv2d para Grad-CAM.")
    return last_name, last_module


def make_gradcam_heatmap(model: torch.nn.Module, image_batch: torch.Tensor, layer_name: str | None = None):
    activations = {}
    gradients = {}

    if layer_name is None:
        layer_name, target_module = find_last_conv_layer(model)
    else:
        modules = dict(model.named_modules())
        target_module = modules[layer_name]

    def forward_hook(_module, _inputs, output):
        activations[layer_name] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradients[layer_name] = grad_output[0].detach()

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        predictions = model(image_batch)
        class_index = int(predictions[0].argmax().item())
        score = predictions[:, class_index].sum()
        score.backward()

        conv_outputs = activations[layer_name][0]
        grads = gradients[layer_name][0]
        pooled_grads = grads.mean(dim=(1, 2))
        heatmap = (conv_outputs * pooled_grads[:, None, None]).sum(dim=0)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy(), class_index
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_gradcam(
    model_path: Path,
    image_path: str,
    experiment_name: str,
    image_size: tuple[int, int] = (224, 224),
) -> Path:
    model, config = _load_checkpoint(model_path)
    image = load_image_for_model(image_path, image_size)
    heatmap, pred_idx = make_gradcam_heatmap(model, image.unsqueeze(0))

    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, image_size)
    heatmap_resized = cv2.resize(heatmap, image_size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * original + 0.45 * colored)

    output_dir = EXPLAINABILITY_FIGURES_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gradcam_{Path(image_path).stem}.png"
    plt.figure(figsize=(8, 3))
    for idx, img in enumerate([original, heatmap_resized, overlay]):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(img, cmap="jet" if idx == 1 else None)
        plt.axis("off")
    plt.suptitle(f"Grad-CAM pred: {RAFDB_LABEL_MAP.get(pred_idx + 1, pred_idx + 1)}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_saliency_map(
    model_path: Path,
    image_path: str,
    experiment_name: str,
    image_size: tuple[int, int] = (224, 224),
) -> Path:
    model, _ = _load_checkpoint(model_path)
    image = load_image_for_model(image_path, image_size).unsqueeze(0)
    image.requires_grad_(True)
    prediction = model(image)
    class_index = prediction.argmax(dim=1)
    loss = prediction[:, class_index.item()].sum()
    loss.backward()
    saliency = image.grad.detach().abs().max(dim=1)[0].squeeze(0).cpu().numpy()
    saliency = saliency / (saliency.max() + 1e-8)

    output_dir = EXPLAINABILITY_FIGURES_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"saliency_{Path(image_path).stem}.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(saliency, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_feature_maps(
    model_path: Path,
    image_path: str,
    experiment_name: str,
    layer_name: str,
    image_size: tuple[int, int] = (224, 224),
    max_maps: int = 16,
) -> Path:
    model, _ = _load_checkpoint(model_path)
    image = load_image_for_model(image_path, image_size).unsqueeze(0)
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"No se encontro la capa {layer_name}.")

    features = {}

    def forward_hook(_module, _inputs, output):
        features["value"] = output.detach()

    handle = modules[layer_name].register_forward_hook(forward_hook)
    try:
        model(image)
        feature_maps = features["value"][0]
        total = min(max_maps, feature_maps.shape[0])

        output_dir = EXPLAINABILITY_FIGURES_DIR / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"feature_maps_{Path(image_path).stem}_{layer_name}.png"
        cols = 4
        rows = int(np.ceil(total / cols))
        plt.figure(figsize=(cols * 2, rows * 2))
        for idx in range(total):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(feature_maps[idx].cpu().numpy(), cmap="viridis")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    finally:
        handle.remove()

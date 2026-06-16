from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm


DATA_ROOT = Path("/data/stepdown vision/data/imagenette/imagenette2-320")
CACHE_ROOT = Path("/data/stepdown vision/cache/perturbations/imagenette/fixed_composition")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 256
SEED = 42
SPLITS = ("train", "val")

# Fixed perturbation strengths. These are meant to be recoverable from the
# additive decomposition families, not destructive enough to collapse identity.
BLUR_SIGMA = 2.0
NOISE_STD = 0.04
JITTER_BRIGHTNESS = 0.85
JITTER_CONTRAST = 0.85
JITTER_SATURATION = 0.60
MASK_RATIO = 0.22
MASK_FILL = 0.50


def odd_kernel_from_sigma(sigma: float) -> int:
    kernel = max(3, int(2 * round(3 * sigma) + 1))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
    return TF.to_tensor(image)


def iter_samples(data_root: Path, split: str) -> list[tuple[Path, str, int]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    samples = []
    for label, class_name in enumerate(classes):
        class_root = split_root / class_name
        for path in sorted(class_root.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_name, label))

    if not samples:
        raise FileNotFoundError(f"No images found under {split_root}")
    return samples


def apply_gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = odd_kernel_from_sigma(sigma)
    return gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def apply_gaussian_noise(image: torch.Tensor, std: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
    return (image + std * noise).clamp(0.0, 1.0)


def apply_color_jitter(
    image: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
) -> torch.Tensor:
    jittered = TF.adjust_brightness(image, brightness)
    jittered = TF.adjust_contrast(jittered, contrast)
    jittered = TF.adjust_saturation(jittered, saturation)
    return jittered.clamp(0.0, 1.0)


def apply_center_mask(image: torch.Tensor, ratio: float, fill_value: float) -> torch.Tensor:
    masked = image.clone()
    _, height, width = masked.shape
    mask_h = max(1, int(height * ratio))
    mask_w = max(1, int(width * ratio))
    top = (height - mask_h) // 2
    left = (width - mask_w) // 2
    masked[:, top : top + mask_h, left : left + mask_w] = fill_value
    return masked


def perturb_image(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    perturbed = apply_gaussian_blur(image, sigma=BLUR_SIGMA)
    perturbed = apply_gaussian_noise(perturbed, std=NOISE_STD, generator=generator)
    perturbed = apply_color_jitter(
        perturbed,
        brightness=JITTER_BRIGHTNESS,
        contrast=JITTER_CONTRAST,
        saturation=JITTER_SATURATION,
    )
    perturbed = apply_center_mask(perturbed, ratio=MASK_RATIO, fill_value=MASK_FILL)
    return perturbed.clamp(0.0, 1.0)


def output_path(cache_root: Path, data_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(data_root).with_suffix(".pt")
    return cache_root / relative


def perturbation_metadata() -> dict:
    return {
        "name": "fixed_composition",
        "pipeline": [
            {"op": "gaussian_blur", "sigma": BLUR_SIGMA},
            {"op": "gaussian_noise", "std": NOISE_STD},
            {
                "op": "color_jitter",
                "brightness": JITTER_BRIGHTNESS,
                "contrast": JITTER_CONTRAST,
                "saturation": JITTER_SATURATION,
            },
            {"op": "center_mask", "mask_ratio": MASK_RATIO, "fill_value": MASK_FILL},
        ],
    }


def precompute():
    generator = torch.Generator().manual_seed(SEED)

    for split in SPLITS:
        samples = iter_samples(DATA_ROOT, split)
        progress = tqdm(samples, desc=f"fixed_composition {split}", unit="image")
        for image_path, class_name, label in progress:
            destination = output_path(CACHE_ROOT, DATA_ROOT, image_path)
            if destination.exists():
                continue

            image = load_image(image_path)
            perturbed = perturb_image(image, generator)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "image": image,
                    "perturbed_image": perturbed,
                    "label": label,
                    "class_name": class_name,
                    "source_path": str(image_path),
                    "perturbation": perturbation_metadata(),
                },
                destination,
            )


if __name__ == "__main__":
    precompute()

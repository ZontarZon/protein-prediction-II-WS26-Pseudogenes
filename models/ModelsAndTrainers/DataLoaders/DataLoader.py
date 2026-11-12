from pathlib import Path
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):

    IMAGE_EXTENSIONS = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.webp",
    )

    def __init__(self, root_dir, image_size):
        """
        root_dir:
            Directory containing the images.

        image_size:
            Target size as (width, height).
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        self.image_paths = self._find_images()

    def _find_images(self):
        """Find all images inside root_dir."""
        images = []

        for extension in self.IMAGE_EXTENSIONS:
            images.extend(
                glob(str(self.root_dir / "**" / extension), recursive=True)
            )

        return sorted(images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")

        image = self._resize_and_pad(image)

        # Convert PIL image -> Tensor
        image = torch.from_numpy(
            __import__("numpy").array(image)
        ).float()

        # HWC -> CHW
        image = image.permute(2, 0, 1)

        # Normalize from [0, 255] -> [-1, 1]
        image = image / 127.5 - 1.0

        return image

    def _resize_and_pad(self, image):
        target_width, target_height = self.image_size

        width, height = image.size

        # Scale image while preserving aspect ratio
        scale = min(
            target_width / width,
            target_height / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        # Create black canvas
        canvas = Image.new(
            "RGB",
            (target_width, target_height),
            (0, 0, 0)
        )

        # Center image
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2

        canvas.paste(image, (x, y))

        return canvas

    def get_min_max_dimensions(self):
        """
        Scan image dimensions without loading image pixel data.

        Returns:
            (min_width, min_height), (max_width, max_height)
        """
        min_width = float("inf")
        min_height = float("inf")

        max_width = 0
        max_height = 0

        for path in self.image_paths:
            with Image.open(path) as image:
                width, height = image.size

            min_width = min(min_width, width)
            min_height = min(min_height, height)

            max_width = max(max_width, width)
            max_height = max(max_height, height)

        if not self.image_paths:
            return None, None

        return (
            (min_width, min_height),
            (max_width, max_height)
        )

    def save_processed_images(self, save_dir):
        """
        Load, resize, pad and save all images to save_dir.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for path in self.image_paths:
            image = Image.open(path).convert("RGB")
            image = self._resize_and_pad(image)

            output_path = save_dir / Path(path).name
            image.save(output_path)

import os
import csv
from collections import defaultdict
from typing import List, Tuple, Dict

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Flickr8kRefDataset(Dataset):
    """Lightweight reference dataset for distillation based on Flickr8k.

    Expects a folder layout:
      root/
        images/  # may contain subfolders with images
        captions.txt  # CSV with header: image,caption

    For simplicity, we map each unique image to a unique integer label and
    keep the first caption as its associated text prompt.
    """
    def __init__(self, root: str, transform=None, num_samples: int = 2048) -> None:
        """
        Args:
            root: 数据集根目录
            transform: torchvision transforms
            num_samples: 如果指定，只收集这么多图像(按排序后的顺序)
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(root, "images")
        self.captions_file = os.path.join(root, "captions.txt")

        captions_map: Dict[str, List[str]] = defaultdict(list)
        if os.path.isfile(self.captions_file):
            with open(self.captions_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    img, cap = row[0].strip(), row[1].strip()
                    captions_map[img].append(cap)

        img_paths: List[str] = []
        for dirpath, _, filenames in os.walk(self.images_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS and fn in captions_map:
                    img_paths.append(os.path.join(dirpath, fn))

        # Sort for determinism
        img_paths.sort()

        # 如果指定了 num_samples，则裁剪
        if num_samples is not None:
            img_paths = img_paths[:num_samples]

        self.samples: List[Tuple[str, int]] = []
        self.prompts_list: List[List[str]] = []
        for idx, path in enumerate(img_paths):
            fname = os.path.basename(path)
            self.samples.append((path, idx))
            self.prompts_list.append(captions_map.get(fname))

        self.num_images = len(self.samples)

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def return_labels_and_prompts(self):
        labels = list(range(len(self.prompts_list)))
        return labels, self.prompts_list

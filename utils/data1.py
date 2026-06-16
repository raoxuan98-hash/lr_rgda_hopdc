# In[]
import os
from typing import List, Optional, Sequence, Callable
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

ROOT = "/data1/open_datasets/chinese-clip-eval/elevater"

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def underline_to_space(s: str) -> str:
    return s.replace("_", " ")

basic_templates = [
    lambda c: f"a photo of the {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"a low contrast photo of a {c}.",
    lambda c: f"a high contrast photo of a {c}.",
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a photo of a big {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a low contrast photo of the {c}.",
    lambda c: f"a high contrast photo of the {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the big {c}."]

class SimpleDataset(Dataset):
    """返回 (PIL.Image, label, class_name)，可选 transform"""
    def __init__(self,
                 images,
                 labels,
                 use_path=False,
                 class_names=None,
                 templates=None,
                 transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.use_path = use_path
        self.class_names = class_names
        self.templates = templates
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = pil_loader(self.images[idx])
        else:
            image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])
        class_name = self.class_names[label] if self.class_names is not None else None
        return image, label, class_name


import numpy as np

class BaseData:
    """只负责提供原始数据，不做 transform"""
    def __init__(self, dataset_name: Optional[str] = None) -> None:
        self.dataset_name = dataset_name
        self.use_path: bool = False
        self.class_order: Optional[Sequence[int]] = None
        self.class_names: Optional[List[str]] = None
        self.train_data: Optional[np.ndarray] = None
        self.train_targets: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        self.test_targets: Optional[np.ndarray] = None
        self.templates: Optional[List[Callable]] = None

        # 控制参数
        self.limit_test_samples: bool = False
        self.max_test_samples: Optional[int] = 2048
        self.shuffle_test_samples: bool = True

    def download_data(self) -> None:
        raise NotImplementedError

    def process_labels(self):
        if self.class_names is not None:
            self.class_names = [underline_to_space(x) for x in self.class_names]

    def build_dataset(self, train=True, transform=None):
        """返回一个 SimpleDataset，可选限制+随机测试集样本数"""
        if train:
            data = self.train_data
            labels = self.train_targets
        else:
            data = self.test_data
            labels = self.test_targets
            if self.limit_test_samples and self.max_test_samples is not None:
                n = len(data)
                if self.shuffle_test_samples:
                    idx = np.random.permutation(n)[:self.max_test_samples]
                else:
                    idx = np.arange(self.max_test_samples)
                data = data[idx]
                labels = labels[idx]

        return SimpleDataset(data, labels,
                             use_path=self.use_path,
                             class_names=self.class_names,
                             templates=self.templates,
                             transform=transform)


class CIFAR100_224(BaseData):
    def __init__(self) -> None:
        super().__init__()
        self.class_order = list(range(100))
        self.download_data()

    def download_data(self) -> None:
        train_dataset = datasets.CIFAR100(ROOT, train=True, download=True)
        test_dataset = datasets.CIFAR100(ROOT, train=False, download=True)
        self.train_data = train_dataset.data
        self.train_targets = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_targets = np.array(test_dataset.targets)
        self.class_names = list(train_dataset.classes)
        self.process_labels()


class ImageFolder224(BaseData):
    def __init__(self, dataset_name: str, templates: Optional[List[Callable]] = None) -> None:
        super().__init__(dataset_name=dataset_name)
        self.use_path = True
        self.templates = templates or basic_templates
        self.download_data()

    def download_data(self) -> None:
        if self.dataset_name is None:
            raise ValueError("Dataset name must be provided and cannot be None.")
        dataset_root = os.path.join(ROOT, self.dataset_name)
        train_dir = os.path.join(dataset_root, "train")
        test_dir = os.path.join(dataset_root, "test")
        label_file = os.path.join(dataset_root, "label.txt")

        if not os.path.exists(train_dir):
            raise RuntimeError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise RuntimeError(f"Test directory not found: {test_dir}")

        train_ds = datasets.ImageFolder(train_dir)
        test_ds = datasets.ImageFolder(test_dir)

        paths, labels = zip(*train_ds.imgs)
        self.train_data = np.array(paths)
        self.train_targets = np.array(labels)
        paths, labels = zip(*test_ds.imgs)
        self.test_data = np.array(paths)
        self.test_targets = np.array(labels)

        # 尝试从 label.txt 读取类名，如果不存在则从 ImageFolder 获取
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # 从 ImageFolder 获取类名，注意：train_ds 和 test_ds 应该有相同的类
            self.class_names = train_ds.classes
            
        self.class_order = list(range(len(self.class_names)))
        self.process_labels()


class Caltech101_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a painting of a {c}.",
            lambda c: f"a plastic {c}.",
            lambda c: f"a sculpture of a {c}.",
            lambda c: f"a sketch of a {c}.",
            lambda c: f"a tattoo of a {c}.",
            lambda c: f"a toy {c}.",
            lambda c: f"a rendition of a {c}.",
            lambda c: f"a embroidered {c}.",
            lambda c: f"a cartoon {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"a plushie {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"art of a {c}.",
            lambda c: f"graffiti of a {c}.",
            lambda c: f"a drawing of a {c}.",
            lambda c: f"a doodle of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a painting of the {c}.",
            lambda c: f"the plastic {c}.",
            lambda c: f"a sculpture of the {c}.",
            lambda c: f"a sketch of the {c}.",
            lambda c: f"a tattoo of the {c}.",
            lambda c: f"the toy {c}.",
            lambda c: f"a rendition of the {c}.",
            lambda c: f"the embroidered {c}.",
            lambda c: f"the cartoon {c}.",
            lambda c: f"the {c} in a video game.",
            lambda c: f"the plushie {c}.",
            lambda c: f"the origami {c}.",
            lambda c: f"art of the {c}.",
            lambda c: f"graffiti of the {c}.",
            lambda c: f"a drawing of the {c}.",
            lambda c: f"a doodle of the {c}."]
        
        super().__init__("caltech-101", templates)


class DTD_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [
            lambda c: f'a photo of a {c} texture.',
            lambda c: f'a photo of a {c} pattern.',
            lambda c: f'a photo of a {c} thing.',
            lambda c: f'a photo of a {c} object.',
            lambda c: f'a photo of the {c} texture.',
            lambda c: f'a photo of the {c} pattern.',
            lambda c: f'a photo of the {c} thing.',
            lambda c: f'a photo of the {c} object.']
        super().__init__("dtd", templates)

class EuroSAT_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [
            lambda c: f"a centered satellite photo of {c}.",
            lambda c: f"a centered satellite photo of a {c}.",
            lambda c: f"a centered satellite photo of the {c}."]
        super().__init__("eurosat_clip", templates)


class Aircraft_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [
            lambda c: f"a photo of a {c}, a type of aircraft.",
            lambda c: f"a photo of the {c}, a type of aircraft.",
        ]
        super().__init__("fgvc-aircraft-2013b-variants102", templates)


class Food101_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [lambda c: f"a photo of a {c}, a type of food."]
        super().__init__("food-101", templates)


class MNIST_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [lambda c: f'a photo of the number: "{c}".']
        super().__init__("mnist", templates)


class OxfordFlower102_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [lambda c: f"a photo of a {c}, a type of flower."]
        super().__init__("oxford-flower-102", templates)


class OxfordPets_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [lambda c: f"a photo of a {c}, a type of pet."]
        super().__init__("oxford-iiit-pets", templates)


class StanfordCars_224(ImageFolder224):
    def __init__(self) -> None:
        templates = [
            lambda c: f"a photo of a {c}, a type of car.",
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"i love my {c}!",
            lambda c: f"a photo of my dirty {c}.",
            lambda c: f"a photo of my clean {c}.",
            lambda c: f"a photo of my new {c}.",
            lambda c: f"a photo of my old {c}."]
        super().__init__("cars196", templates)


# 统一工厂方法
DATASET_NAME_TO_CLASS = {
    "cifar100_224": CIFAR100_224,
    "caltech-101": Caltech101_224,
    "dtd": DTD_224,
    "eurosat": EuroSAT_224,
    "eurosat_clip": EuroSAT_224,
    "fgvc_aircraft": Aircraft_224,
    "fgvc-aircraft-2013b-variants102": Aircraft_224,
    "aircraft": Aircraft_224,
    "food101": Food101_224,
    "food-101": Food101_224,
    "mnist": MNIST_224,
    "oxford_flower102": OxfordFlower102_224,
    "oxford-flower-102": OxfordFlower102_224,
    "flower102": OxfordFlower102_224,
    "oxford_pets": OxfordPets_224,
    "oxford-iiit-pets": OxfordPets_224,
    "cars196_224": StanfordCars_224,
    "stanford-cars": StanfordCars_224,
    "fer2013": lambda: ImageFolder224("fer-2013"),
    "gtsrb": lambda: ImageFolder224("gtsrb"),
    "hateful_memes": lambda: ImageFolder224("hateful-memes"),
    "imagenet-a": lambda: ImageFolder224("imagenet-a"),
    "imagenet_a": lambda: ImageFolder224("imagenet-a"),
    "imagenet-r": lambda: ImageFolder224("imagenet-r"),
    "kitti_distance": lambda: ImageFolder224("kitti-distance"),
    "patch_camelyon": lambda: ImageFolder224("patch-camelyon"),
    "rendered_sst2": lambda: ImageFolder224("rendered-sst2"),
    "resisc45": lambda: ImageFolder224("resisc45_clip"),
    "vtab": lambda: ImageFolder224("vtab"),
    "cub200_224": lambda: ImageFolder224("cub_200"),
}


def get_dataset(name: str):
    """工厂函数，根据名字返回对应类实例"""
    if name not in DATASET_NAME_TO_CLASS:
        raise ValueError(f"Unknown dataset {name}")
    cls_or_func = DATASET_NAME_TO_CLASS[name]
    return cls_or_func() if callable(cls_or_func) else cls_or_func()

# In[]
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    import os

    # ==================== CLIP Zero-Shot ====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch16"

    print(f"Loading CLIP model: {model_name}")
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def collate_fn(batch):
        # batch: [(image, label, class_name), ...]
        images, labels, class_names = zip(*batch)  # -> tuple
        # 关键：转成 list
        proc = processor(images=list(images), return_tensors="pt")
        pixel_values = proc["pixel_values"]
        labels = torch.tensor(labels, dtype=torch.long)
        class_names = list(class_names)
        return pixel_values, labels, class_names
        
    # CIFAR100 数据（你已有 CIFAR100_224 类）
    dataset = get_dataset("dtd")
    test_dataset = dataset.build_dataset(train=False, transform=None)

    test_loader = DataLoader(
        test_dataset, batch_size=64,
        num_workers=0, pin_memory=True,
        shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        all_text_features = []
        for name in dataset.class_names:
            # 为当前类别生成所有模板的文本
            class_prompts = [template(name) for template in dataset.templates]
            text_inputs = processor(
                text=class_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            class_text_features = model.get_text_features(**text_inputs)
            class_text_features /= class_text_features.norm(dim=-1, keepdim=True)
            # 对该类别的所有模板取平均
            avg_text_feature = class_text_features.mean(dim=0)
            avg_text_feature /= avg_text_feature.norm()  # 再次归一化
            all_text_features.append(avg_text_feature)
        
        text_features = torch.stack(all_text_features, dim=0)  # (100, D)

    # 遍历测试集
    correct, total = 0, 0
    for images, labels, names in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_features = model.get_image_features(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            preds = torch.argmax(logits, dim=-1)

        labels = labels.to(device)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Zero-Shot Accuracy on CIFAR100_224 ({model_name}): {acc:.4f}")
# %%

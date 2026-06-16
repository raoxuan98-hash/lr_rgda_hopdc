# adapt_backbone 函数在 Cross-Domain 数据集上的问题分析

## 问题描述

在 [`classifier_ablation/experiments/exp6_rank.py`](classifier_ablation/experiments/exp6_rank.py) 中，当 `iterations > 0` 时调用 [`adapt_backbone`](classifier_ablation/features/feature_extractor.py:35) 函数会导致运行失败，而相同的代码在 [`classifier_ablation/experiments/exp6_rank_within_domain.py`](classifier_ablation/experiments/exp6_rank_within_domain.py) 中可以正常运行。

核心原因是：**Cross-Domain 数据集和 Within-Domain 数据集的标签处理方式不同，导致 `adapt_backbone` 函数无法正确处理 Cross-Domain 数据集的标签。**

## 问题根源分析

### 1. Within-Domain 数据集（正常工作）

在 [`exp6_rank_within_domain.py`](classifier_ablation/experiments/exp6_rank_within_domain.py:352) 中：

```python
adapt_loader = DataLoader(train_subsets, batch_size=25, shuffle=True, num_workers=4)
vit = adapt_backbone(vit, adapt_loader, dataset.nb_tasks * (args.init_cls if args.init_cls > 0 else 50), iterations=args.iterations)
```

**Within-Domain 数据集的标签处理：**

1. [`IncrementalDataManager`](utils/data_manager1.py:90) 在 `_remap_all_labels` 方法中将原始标签重新映射到 [0, C-1] 的连续范围：
   ```python
   mapping: Dict[int, int] = {orig: new for new, orig in enumerate(self._class_order)}
   self._train_targets = np.asarray([mapping[int(y)] for y in self._train_targets], dtype=np.int64)
   self._test_targets = np.asarray([mapping[int(y)] for y in self._test_targets], dtype=np.int64)
   ```

2. [`get_subset`](utils/data_manager1.py:138) 方法返回的数据集中，标签已经是重新映射后的连续值 [0, N-1]

3. `total_classes = dataset.nb_tasks * (args.init_cls if args.init_cls > 0 else 50)` 正确反映了实际类别数

**结果：** 标签范围与 `total_classes` 参数一致，`adapt_backbone` 函数正常工作。

### 2. Cross-Domain 数据集（存在问题）

在 [`exp6_rank.py`](classifier_ablation/experiments/exp6_rank.py:290-291) 中：

```python
adapt_loader = create_adapt_loader(train_subsets)
vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=args.iterations)
```

**Cross-Domain 数据集的标签处理存在双重偏移问题：**

#### 第一次偏移（初始化时）

在 [`CrossDomainDataManagerCore.__init__`](utils/cross_domain_data_manager.py:104-113) 中：

```python
# Apply global label offset
offset = self.total_classes

# 应用全局标签偏移到原始标签
dataset_info['train_targets'] = dataset_info['train_targets'] + offset
dataset_info['test_targets'] = dataset_info['test_targets'] + offset

self.datasets.append(dataset_info)
self.global_label_offset.append(offset)
self.total_classes += dataset_info['num_classes']
```

此时，每个数据集的标签已经被添加了全局偏移：
- 第一个数据集（cifar100_224）：offset = 0，标签范围 [0, 99]
- 第二个数据集（imagenet-r）：offset = 100，标签范围 [100, 299]
- 第三个数据集（cars196_224）：offset = 300，标签范围 [300, 495]
- ...

#### 第二次偏移（get_subset 时）

在 [`CrossDomainDataManagerCore.get_subset(cumulative=True)`](utils/cross_domain_data_manager.py:169-204) 中：

```python
if cumulative:
    # 累积模式：返回当前任务与先前数据集的拼接数据集
    all_data = []
    all_targets = []
    use_path = False
    templates = []
    
    for i in range(task + 1):
        dataset = self.datasets[i]
        if source == "train":
            data = dataset['train_data']
            targets = dataset['train_targets']
        else:
            data = dataset['test_data']
            targets = dataset['test_targets']
        
        # 标签已经在初始化时添加了全局偏移，不需要再次添加
        
        targets = targets + self.global_label_offset[i]  # ← 问题：再次添加偏移！
        
        all_data.extend(data)
        all_targets.extend(targets)
        ...
```

**问题：** 第187行 `targets = targets + self.global_label_offset[i]` **再次添加了全局偏移**，导致标签值超出预期范围。

#### 实际问题场景

在 [`exp6_rank.py`](classifier_ablation/experiments/exp6_rank.py:33-34) 中：

```python
subsets = dataset.get_subset(len(cross_domain_datasets) - 1, source='train', cumulative=True, mode="test")
train_subsets, test_subsets = random_split(subsets, [0.5, 0.5])
```

假设 `cross_domain_datasets` 包含7个数据集：
- 数据集0 (cifar100_224): 100类，offset=0
- 数据集1 (imagenet-r): 200类，offset=100
- 数据集2 (cars196_224): 196类，offset=300
- 数据集3 (cub200_224): 200类，offset=496
- 数据集4 (caltech-101): 101类，offset=696
- 数据集5 (oxford-flower-102): 102类，offset=797
- 数据集6 (food-101): 101类，offset=899

**预期行为：**
- `dataset.total_classes` = 1000
- 标签范围应该是 [0, 999]

**实际行为（双重偏移后）：**
- 数据集0的标签：0 + 0 = 0 ~ 99 + 0 = 99
- 数据集1的标签：100 + 100 = 200 ~ 299 + 100 = 399
- 数据集2的标签：300 + 300 = 600 ~ 495 + 300 = 795
- 数据集3的标签：496 + 496 = 992 ~ 695 + 496 = 1191
- 数据集4的标签：696 + 696 = 1392 ~ 796 + 696 = 1492
- 数据集5的标签：797 + 797 = 1594 ~ 898 + 797 = 1695
- 数据集6的标签：899 + 899 = 1798 ~ 999 + 899 = 1898

**结果：** 最大标签值达到 1898，远超 `dataset.total_classes` (1000)。

### 3. adapt_backbone 函数的限制

[`adapt_backbone`](classifier_ablation/features/feature_extractor.py:35-113) 函数期望：

```python
def adapt_backbone(vit, train_loader, total_classes, iterations=0, ...):
    ...
    # 创建分类器
    classifier = nn.Linear(768, total_classes).to(device)  # 输出维度为 total_classes
    criterion = nn.CrossEntropyLoss()
    ...
    while iteration < iterations:
        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)  # 标签应该范围 [0, total_classes-1]
            ...
            outputs = classifier(features)
            loss = criterion(outputs, labels)  # CrossEntropyLoss 期望标签 < total_classes
```

**问题：** 当标签值 ≥ `total_classes` 时，`CrossEntropyLoss` 会抛出错误。

## 解决方案

### 方案1：修复 CrossDomainDataManagerCore.get_subset 方法

移除 [`cross_domain_data_manager.py:187`](utils/cross_domain_data_manager.py:187) 中的重复偏移：

```python
# 修改前
targets = targets + self.global_label_offset[i]

# 修改后
# 标签已经在初始化时添加了全局偏移，不需要再次添加
# targets = targets + self.global_label_offset[i]  # 删除这行
```

### 方案2：在 exp6_rank.py 中重新映射标签

在调用 `adapt_backbone` 之前，对 `train_subsets` 中的标签进行重新映射：

```python
# 在 exp6_rank.py 中添加标签重新映射逻辑
from torch.utils.data import Dataset

class RemappedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
        # 收集所有唯一标签并创建映射
        all_labels = []
        for i in range(len(original_dataset)):
            _, label, _ = original_dataset[i]
            all_labels.append(label)
        
        unique_labels = sorted(set(all_labels))
        self.label_mapping = {old: new for new, old in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        image, label, class_name = self.original_dataset[idx]
        return image, self.label_mapping[label], class_name

# 使用重新映射的数据集
remapped_train_subsets = RemappedDataset(train_subsets)
adapt_loader = create_adapt_loader(remapped_train_subsets)
vit = adapt_backbone(vit, adapt_loader, remapped_train_subsets.num_classes, iterations=args.iterations)
```

### 方案3：修改 adapt_backbone 函数以支持非连续标签

修改 [`adapt_backbone`](classifier_ablation/features/feature_extractor.py:35) 函数，使其能够处理非连续标签：

```python
def adapt_backbone(vit, train_loader, total_classes, iterations=0, 
                   vit_lr=1e-5, classifier_lr=1e-3, ema_beta=0.90):
    ...
    # 创建分类器
    classifier = nn.Linear(768, total_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 添加标签重新映射逻辑
    all_labels = []
    for batch in train_loader:
        all_labels.append(batch[1].cpu().numpy())
    all_labels = np.concatenate(all_labels)
    unique_labels = sorted(set(all_labels))
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    
    def remap_labels(labels):
        return torch.tensor([label_mapping[l.item()] for l in labels]).to(device)
    
    ...
    while iteration < iterations:
        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = remap_labels(batch[1])  # 重新映射标签
            ...
```

## 推荐方案

**推荐使用方案1**，因为：

1. 问题根源在于 [`CrossDomainDataManagerCore.get_subset`](utils/cross_domain_data_manager.py:151) 方法的实现错误
2. 修复这个方法可以解决所有使用该方法的代码中的问题
3. 不需要在每个实验脚本中添加额外的标签处理逻辑
4. 保持代码的一致性和可维护性

## 影响范围

这个问题可能影响所有使用 `CrossDomainDataManagerCore` 或 `BalancedCrossDomainDataManagerCore` 且调用 `get_subset(cumulative=True)` 的实验脚本，包括但不限于：

- [`classifier_ablation/experiments/exp6_rank.py`](classifier_ablation/experiments/exp6_rank.py)
- 其他使用 cross-domain 数据集且需要 `adapt_backbone` 的实验

## 相关文件

- [`classifier_ablation/experiments/exp6_rank.py`](classifier_ablation/experiments/exp6_rank.py) - Cross-Domain 实验脚本（存在问题）
- [`classifier_ablation/experiments/exp6_rank_within_domain.py`](classifier_ablation/experiments/exp6_rank_within_domain.py) - Within-Domain 实验脚本（正常工作）
- [`classifier_ablation/features/feature_extractor.py`](classifier_ablation/features/feature_extractor.py:35) - `adapt_backbone` 函数实现
- [`utils/cross_domain_data_manager.py`](utils/cross_domain_data_manager.py) - Cross-Domain 数据管理器（存在双重偏移问题）
- [`utils/balanced_cross_domain_data_manager.py`](utils/balanced_cross_domain_data_manager.py) - 平衡 Cross-Domain 数据管理器
- [`utils/data_manager1.py`](utils/data_manager1.py) - Within-Domain 数据管理器（正常工作）
- [`classifier_ablation/data/data_loader.py`](classifier_ablation/data/data_loader.py) - 数据加载器

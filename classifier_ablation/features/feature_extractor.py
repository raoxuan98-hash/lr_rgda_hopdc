"""
特征提取模块
"""
import os
import torch
import torch.nn as nn
import timm
import tqdm

def get_vit(vit_name="vit-b-p16"):
    name = vit_name.lower()
    
    if name == 'vit-b-p16':
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    elif name == 'vit-b-p16-mocov3':
        vit = timm.create_model('vit_base_patch16_224.', pretrained=False, num_classes=0)
        model_dict = torch.load('mocov3-vit-base-300ep.pth', weights_only=False)
        vit.load_state_dict(model_dict['model'], strict=True)
    elif name == 'vit-b-p16-dino':
        vit = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)
    elif name == 'vit-b-p16-mae':
        vit = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0)
    elif name == 'vit-b-p16-clip':
        vit = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0)
    else:
        raise ValueError(f'Model {name} not supported')
    
    vit.head = nn.Identity()

    if name == 'vit-b-p16-mocov3':
        del vit.norm
        vit.norm = nn.LayerNorm(768, elementwise_affine=False)
    return vit

def adapt_backbone(vit, train_loader, total_classes, iterations=0, 
                   vit_lr=1e-5, classifier_lr=1e-3, ema_beta=0.90):
    """
    适应网络主干
    
    Args:
        vit: ViT模型
        train_loader: 训练数据加载器
        total_classes: 总类别数
        iterations: 训练迭代次数
        vit_lr: ViT学习率
        classifier_lr: 分类器学习率
        ema_beta: EMA参数
    
    Returns:
        vit: 适应后的ViT模型
    """
    if iterations <= 0:
        return vit
    
    print("开始适应网络主干...")
    
    device = "cuda"
    vit.to(device)
    
    # 创建分类器
    classifier = nn.Linear(768, total_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 设置优化器
    optimizer = torch.optim.AdamW([
        {'params': vit.parameters(), 'lr': vit_lr},
        {'params': classifier.parameters(), 'lr': classifier_lr}
    ])
    
    # EMA参数
    ema_loss = 0.0
    ema_acc = 0.0
    iteration = 0
    
    vit.train()
    classifier.train()
    
    while iteration < iterations:
        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            
            optimizer.zero_grad()
            features = vit(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 计算当前batch准确率
            pred = outputs.argmax(dim=1)
            acc = pred.eq(labels).float().mean().item()
            loss_val = loss.item()
            
            if iteration == 0:
                ema_loss = loss_val
                ema_acc = acc
            else:
                ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_val
                ema_acc = ema_beta * ema_acc + (1 - ema_beta) * acc
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, EMA Loss: {ema_loss:.4f}, EMA Acc: {ema_acc:.4f}")
            
            iteration += 1
            
            if iteration >= iterations:
                break
    
    print("网络主干适应完成")
    vit.eval()
    
    return vit

def extract_features_and_labels(model, dataset, train_loader, test_loader, model_name, 
                               num_shots, iterations=None, cache_dir="cached_data/classifier_ablation"):
    # 创建缓存键
    if iterations is not None:
        cache_key = f"{model_name}_{num_shots}_iter{iterations}_features_cache"
    else:
        cache_key = f"{model_name}_{num_shots}_features_cache"
    
    cache_path = os.path.join(cache_dir, cache_key)
    
    # 检查缓存
    cache_files = [
        f"{cache_path}_train_features.pt",
        f"{cache_path}_train_labels.pt",
        f"{cache_path}_train_dataset_ids.pt",
        f"{cache_path}_test_features.pt",
        f"{cache_path}_test_labels.pt",
        f"{cache_path}_test_dataset_ids.pt"
    ]
    
    if all(os.path.exists(f) for f in cache_files):
        print(f"检测到缓存文件，直接加载特征和标签...")
        print(f"缓存键: {cache_key}")
        
        train_features = torch.load(cache_files[0])
        train_labels = torch.load(cache_files[1])
        train_dataset_ids = torch.load(cache_files[2])
        test_features = torch.load(cache_files[3])
        test_labels = torch.load(cache_files[4])
        test_dataset_ids = torch.load(cache_files[5])
        
        print(f"缓存加载完成:")
        print(f"  训练特征: {train_features.shape}")
        print(f"  训练标签: {train_labels.shape}")
        print(f"  测试特征: {test_features.shape}")
        print(f"  测试标签: {test_labels.shape}")
        
        return (train_features, train_labels, train_dataset_ids,
                test_features, test_labels, test_dataset_ids)
    
    print("未检测到缓存，开始提取特征...")
    
    model.eval()
    device = "cuda"
    model.to(device)
    
    # 提取训练特征
    print("提取训练特征...")
    train_features = []
    train_targets = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader):
            inputs = batch[0].to(device)
            labels = batch[1]
            feats = model(inputs).cpu()
            train_features.append(feats)
            train_targets.append(labels.cpu())
    
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_targets, dim=0)
    
    # 提取测试特征
    print("提取测试特征...")
    test_features = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            inputs = batch[0].to(device)
            labels = batch[1]
            feats = model(inputs).cpu()
            test_features.append(feats)
            test_targets.append(labels.cpu())
    
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_targets, dim=0)
    
    # 对于 within_domain 数据集，所有样本都来自同一个数据集，所以数据集ID都设为0
    train_dataset_ids = torch.zeros_like(train_labels)
    test_dataset_ids = torch.zeros_like(test_labels)

    # 保存缓存
    print("保存特征缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    
    torch.save(train_features, cache_files[0])
    torch.save(train_labels, cache_files[1])
    torch.save(train_dataset_ids, cache_files[2])
    torch.save(test_features, cache_files[3])
    torch.save(test_labels, cache_files[4])
    torch.save(test_dataset_ids, cache_files[5])
    
    print(f"缓存已保存到: {cache_dir}")
    
    return (train_features, train_labels, train_dataset_ids,
            test_features, test_labels, test_dataset_ids)

def infer_dataset_ids_from_labels(labels, dataset_manager):
    """
    从标签推断数据集ID
    
    Args:
        labels: 标签张量
        dataset_manager: 数据集管理器
    
    Returns:
        dataset_ids: 数据集ID列表
    """
    dataset_ids = []
    
    label_ranges = []
    for i in range(len(dataset_manager.datasets)):
        offset = dataset_manager.global_label_offset[i]
        num_classes = dataset_manager.datasets[i]['num_classes']
        label_ranges.append((offset, offset + num_classes - 1))
    
    # 为每个标签推断数据集ID
    for label in labels:
        label_item = label.item()
        for i, (start, end) in enumerate(label_ranges):
            if start <= label_item <= end:
                dataset_ids.append(i)
                break
        else:
            dataset_ids.append(0)
    
    return dataset_ids
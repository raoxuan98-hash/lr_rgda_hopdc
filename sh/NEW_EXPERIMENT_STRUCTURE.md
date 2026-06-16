# 实验脚本重构说明

## 概述

根据实验需求，我们已经重构了所有实验脚本，正确分离了cross-domain和within-domain实验，并优化了GPU分配策略。

## 新的脚本结构

### 1. 域内实验 (Within-Domain Experiments)

**脚本**: [`sh/within_domain_experiments.sh`](sh/within_domain_experiments.sh)

**数据集**:
- cifar100_224
- imagenet-r
- cub200_224
- cars196_224

**GPU分配策略**:
- 每个数据集分配一个GPU (GPU 0, 1, 2, 4)
- 同一数据集的不同随机种子顺序运行

**实验变体** (8种):
1. **basic_lora**: 基础LoRA (gamma_kd=0.0)
2. **lora_kd_1.0**: LoRA + 知识蒸馏 (gamma_kd=1.0)
3. **lora_kd_0.5**: LoRA + 知识蒸馏 (gamma_kd=0.5)
4. **nsp_lora_0.05**: LoRA-NSP (nsp_weight=0.05)
5. **nsp_lora_0.00**: LoRA-NSP (nsp_weight=0.00)
6. **sgp_lora_t1.0_p1.0**: LoRA-SGP (weight_temp=1.0, weight_p=1.0)
7. **sgp_lora_t2.0_p1.0**: LoRA-SGP (weight_temp=2.0, weight_p=1.0)
8. **sgp_lora_t2.0_p2.0**: LoRA-SGP (weight_temp=2.0, weight_p=2.0)

**运行命令**:
```bash
bash sh/within_domain_experiments.sh
```

### 2. 跨域实验 (Cross-Domain Experiments)

**脚本**: [`sh/cross_domain_experiments_new.sh`](sh/cross_domain_experiments_new.sh)

**数据集**:
- cross_domain_elevater (固定数据集序列)

**GPU分配策略**:
- 每个随机种子分配一个GPU (GPU 0, 1, 2)
- 不同随机种子可以同时并行运行

**实验变体** (7种):
1. **basic_lora**: 基础LoRA (gamma_kd=0.0)
2. **lora_kd_1.0**: LoRA + 知识蒸馏 (gamma_kd=1.0)
3. **lora_kd_0.5**: LoRA + 知识蒸馏 (gamma_kd=0.5)
4. **nsp_lora_0.05**: LoRA-NSP (nsp_weight=0.05)
5. **nsp_lora_0.00**: LoRA-NSP (nsp_weight=0.00)
6. **sgp_lora_t2.0_p1.0**: LoRA-SGP (weight_temp=2.0, weight_p=1.0)
7. **sgp_lora_t2.0_p2.0**: LoRA-SGP (weight_temp=2.0, weight_p=2.0)

**跨域实验特定参数**:
- cross_domain=True
- num_shots=16

**运行命令**:
```bash
bash sh/cross_domain_experiments_new.sh
```

### 3. 总控脚本

**脚本**: [`sh/run_all_experiments.sh`](sh/run_all_experiments.sh)

**功能**: 依次运行所有域内和跨域实验

**运行命令**:
```bash
bash sh/run_all_experiments.sh
```

## 实验数量统计

### 域内实验
- 数据集数量: 4
- 随机种子: 3 (1993, 1996, 1997)
- 实验变体: 8
- 总实验数: 4 × 3 × 8 = 96

### 跨域实验
- 数据集: 1 (cross_domain_elevater)
- 随机种子: 3 (1993, 1996, 1997)
- 实验变体: 7
- 总实验数: 1 × 3 × 7 = 21

### 总计
- 总实验数: 96 + 21 = 117

## GPU使用优化

### 域内实验
- 最大并行度: 4 (每个数据集一个GPU)
- 每个GPU上顺序运行3个种子

### 跨域实验
- 最大并行度: 3 (每个种子一个GPU)
- 所有种子可以完全并行运行

## 日志存储结构

实验结果将根据以下结构存储:
```
logs/
├── within_domain_experiments_YYYYMMDD_HHMMSS/
│   ├── {experiment_type}_YYYYMMDD_HHMMSS/
│   │   ├── run_{dataset}.sh
│   │   ├── {dataset}_seed{seed}.log
│   │   └── ...
│   └── ...
└── cross_domain_experiments_YYYYMMDD_HHMMSS/
    ├── {experiment_type}_YYYYMMDD_HHMMSS/
    │   ├── run_seed_{seed}.sh
    │   ├── seed{seed}.log
    │   └── ...
    └── ...
```

## 参数配置详解

### 基础参数
- `--smart_defaults`: 根据数据集自动调整参数
- `--vit_type`: 使用vit-b-p16-mocov3架构
- `--seed_list`: 随机种子 (1993, 1996, 1997)

### LoRA类型特定参数

#### 基础LoRA
- `--lora_type`: basic_lora
- `--gamma_kd`: 0.0 (不使用知识蒸馏)

#### LoRA + 知识蒸馏
- `--lora_type`: basic_lora
- `--gamma_kd`: 1.0 或 0.5
- `--update_teacher_each_task`: True
- `--distillation_transform`: identity
- `--kd_type`: feat

#### LoRA-NSP
- `--lora_type`: nsp_lora
- `--gamma_kd`: 0.0
- `--nsp_weight`: 0.05 或 0.00
- `--nsp_eps`: 0.05

#### LoRA-SGP
- `--lora_type`: sgp_lora
- `--gamma_kd`: 0.0
- `--weight_temp`: 1.0 或 2.0
- `--weight_p`: 1.0 或 2.0
- `--weight_kind`: log1p

## 使用建议

1. **资源规划**: 确保有足够的GPU资源 (域内实验需要4个GPU，跨域实验需要3个GPU)
2. **存储空间**: 实验会产生大量日志文件，确保有足够的存储空间
3. **运行时间**: 完整实验可能需要数小时到数天
4. **监控**: 建议在实验过程中定期检查日志文件
5. **结果分析**: 实验完成后，可以使用提供的统计脚本分析结果

## 故障排除

如果遇到问题，请检查:
1. GPU是否可用且未被其他进程占用
2. 数据集路径是否正确配置
3. 依赖库是否完整安装
4. 脚本权限是否正确设置
5. 日志目录是否有写入权限
## ConvST

在此项目中复现了ViT、CvT两篇论文，并基于CvT改进得到CCvT、CSvT两个算法，并尝试在CIFAR-10数据集训练。

## 使用

### 1. 环境

python=3.8.13, CUDA=11.7

```bash
pip install -r  requirement.txt
```

### 2. 训练

```bash
python train.py-net vit （-gpu）
```

### 3. 验证

```bash
python test.py -net vit -weights path_to_the_weight
```

### 4. 训练后模型

见checkpoint目录

## 参考

https://github.com/lucidrains/vit-pytorch

https://github.com/weiaicunzai/pytorch-cifar100

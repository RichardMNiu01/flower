## 1. 项目整体目标

本项目是一个花卉图片分类系统。输入一张花的图片，模型输出它最可能属于哪一种花。

当前项目包含 10 个类别，类别名称保存在 `classes.txt` 中：

```text
Bellflower
Carnation
Daisy
Dandelion
Lavender
Lily
Lotus
Rose
Sunflower
Tulip
```

也就是说，模型最后需要从这 10 类花中选出一个预测结果。

## 2. 项目目录结构

项目中的关键文件和目录如下：

```text
FlowerClassify/
├── configs/
│   └── config.toml              # 训练和评估使用的配置文件
├── datasets/
│   ├── train/                   # 训练集
│   ├── valid/                   # 验证集
│   └── test/                    # 测试集
├── checkpoints/
│   ├── best-ckpt2.pt            # 验证集效果最好的 PyTorch 模型权重
│   └── last-ckpt2.pt            # 最后一轮训练保存的 PyTorch 模型权重
├── servers/
│   ├── server.py                # Flask + ONNX Runtime 推理服务
│   ├── configs/config.toml      # 服务端推理配置
│   └── models/flower-fp32.onnx  # ONNX 格式模型
├── classes.txt                  # 类别名称
├── models.py                    # 模型结构定义
├── train.py                     # 模型训练脚本
├── eval.py                      # 模型评估脚本
├── gui.py                       # 本地图形界面预测程序
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明

```

## 3. 数据集组织方式

项目使用 PyTorch 的 `ImageFolder` 读取图片数据。

`ImageFolder` 要求数据按照“一个类别一个文件夹”的形式存放，例如：

```text
datasets/train/
├── Bellflower/
│   ├── xxx.jpg
│   └── ...
├── Carnation/
│   ├── xxx.jpg
│   └── ...
└── ...
```

验证集和测试集也是同样结构：

```text
datasets/valid/
├── Bellflower/
├── Carnation/
└── ...

datasets/test/
├── Bellflower/
├── Carnation/
└── ...
```

这段代码出现在 `train.py`：

```python
train_dataset = ImageFolder('datasets/train', transform=train_transform)
valid_dataset = ImageFolder('datasets/valid', transform=valid_transform)
```

它的作用是：

- 自动扫描 `datasets/train` 和 `datasets/valid`
- 把每个子文件夹当成一个类别
- 给每张图片生成一个数字标签
- 对读取到的图片执行指定的 `transform` 预处理

在 `eval.py` 中也使用了同样的方法读取测试集：

```python
dataset = ImageFolder('datasets/test', transform=transform)
```

## 4. 配置文件 config.toml

训练和评估的主要参数放在 `configs/config.toml` 中：

```toml
device = "cuda"
learning-rate = 0.0002
batch-size = 32
num-epochs = 50
num-workers = 8
num-classes = 10
weight-decay = 0.0001

log-interval = 10
load-checkpoint = false
load-pretrained = true

load-checkpoint-path = "checkpoints/best-ckpt2.pt"
best-checkpoint-path = "checkpoints/best-ckpt2.pt"
last-checkpoint-path = "checkpoints/last-ckpt2.pt"
```

关键字段解释：

| 字段 | 含义 |
| --- | --- |
| `device` | 运行设备，`cuda` 表示使用 NVIDIA GPU，`cpu` 表示使用 CPU |
| `learning-rate` | 学习率，控制模型参数每次更新的步长 |
| `batch-size` | 每次送入模型训练的图片数量 |
| `num-epochs` | 训练轮数，一轮表示完整遍历一次训练集 |
| `num-workers` | 数据加载进程数，用于加速读取图片 |
| `num-classes` | 分类类别数，本项目是 10 类 |
| `weight-decay` | 权重衰减，用于减少过拟合 |
| `log-interval` | 每隔多少个 batch 打印一次训练或验证日志 |
| `load-pretrained` | 是否使用 ImageNet 预训练权重初始化模型 |
| `load-checkpoint` | 是否从已有 checkpoint 继续训练 |
| `load-checkpoint-path` | 加载模型权重的路径 |
| `best-checkpoint-path` | 保存验证集效果最好模型的路径 |
| `last-checkpoint-path` | 保存最后一轮模型的路径 |


## 5. 模型结构 models.py

模型定义在 `models.py` 中，核心类是 `FlowerNet`：

```python
class FlowerNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()

        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18()
```

这里使用的是 ResNet18。

ResNet18 是一个经典的卷积神经网络，原本可以用于 ImageNet 这样的大规模图像分类任务。这个项目借用了 ResNet18 的主干结构，然后把最后的全连接层改成适合 10 类花卉分类的输出。

## 6. train.py 训练流程

`train.py` 是项目的训练入口。

运行命令：

```shell
python train.py
```

训练流程可以分成 8 步：

```text
1. 读取配置
2. 定义训练集和验证集的图片预处理
3. 加载 train 和 valid 数据集
4. 创建 DataLoader
5. 创建模型、损失函数和优化器
6. 进入 epoch 循环训练模型
7. 每轮训练后在验证集上计算准确率
8. 保存 best 和 last 模型权重
```

### 6.1 读取配置

```python
configs = toml.load('configs/config.toml')
```

这行代码会把 `config.toml` 中的配置读取成 Python 字典。

例如：

```python
num_epochs = configs['num-epochs']
```

就是从配置文件中取出训练轮数。

### 6.2 训练集数据增强

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(),
])
```

这些操作的作用：

| 代码 | 作用 |
| --- | --- |
| `Resize((224, 224))` | 把图片统一变成 224x224 |
| `ColorJitter` | 随机调整颜色，让模型适应不同光照 |
| `RandomRotation` | 随机旋转，让模型适应拍摄角度变化 |
| `RandomHorizontalFlip` | 随机水平翻转，提高数据多样性 |
| `ToTensor` | 把图片转成 PyTorch 可以处理的张量 |
| `Normalize` | 标准化像素值，让训练更稳定 |
| `RandomErasing` | 随机遮挡部分区域，提高模型抗遮挡能力 |

为什么训练集要做随机增强？

因为真实环境中的花卉图片可能有不同角度、光照、背景和遮挡。数据增强可以模拟这些变化，让模型不只记住训练图片，而是学会更通用的分类规律。

### 6.3 验证集预处理

```python
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

验证集不使用随机增强。

原因是验证集用于衡量模型效果，如果每次验证时图片都被随机改变，那么验证结果就不稳定，不方便比较不同 epoch 的效果。

### 6.4 加载数据集

```python
train_dataset = ImageFolder('datasets/train', transform=train_transform)
valid_dataset = ImageFolder('datasets/valid', transform=valid_transform)

train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)
```

这几行代码完成两个任务：

- 读取训练集和验证集图片
- 统计训练集和验证集各有多少张图片

### 6.5 创建 DataLoader

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=configs['batch-size'],
    num_workers=configs['num-workers'],
    shuffle=True
)
```

`DataLoader` 的作用是把数据按 batch 送进模型。

假设：

```toml
batch-size = 32
```

那么模型每次训练会同时处理 32 张图片。

训练集使用：

```python
shuffle=True
```

表示每轮训练前打乱图片顺序，这样可以减少模型记住数据排列顺序的风险。

### 6.6 创建模型

```python
device = torch.device(configs['device'])

model = FlowerNet(
    num_classes=configs['num-classes'],
    pretrained=configs['load-pretrained']
)
model = model.to(device)
```

这里做了三件事：

- 根据配置选择运行设备
- 创建 `FlowerNet` 模型
- 把模型移动到 CPU 或 GPU 上

如果 `configs['device']` 是 `"cuda"`，就会使用 GPU 训练。

如果电脑没有 GPU，可以改成：

```toml
device = "cpu"
```

### 6.7 损失函数和优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=configs['learning-rate'],
    weight_decay=configs['weight-decay']
)
```

`CrossEntropyLoss` 是多分类任务中非常常用的损失函数。

它衡量的是：

```text
模型预测结果和真实类别之间差得有多远
```

`Adam` 是优化器，负责根据损失函数计算出的梯度来更新模型参数。

可以把训练过程简单理解成：

```text
模型预测错了
    ↓
损失函数告诉模型错得有多严重
    ↓
反向传播计算每个参数应该怎么改
    ↓
优化器根据学习率更新参数
```

### 6.8 加载 checkpoint

```python
if configs['load-checkpoint']:
    model.load_state_dict(
        torch.load(load_checkpoint_path, map_location=device, weights_only=True)
    )
```

如果 `load-checkpoint = true`，程序会加载已有模型权重继续训练。

适用场景：

- 上次训练中断了，想继续训练
- 想在已有模型基础上再训练几轮
- 想用之前训练好的模型作为初始模型

### 6.9 训练循环

核心训练代码：

```python
for epoch in range(num_epochs):
    model.train()

    for batch, (images, labels) in enumerate(train_dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

这段是整份 `train.py` 最核心的部分。

每个 batch 的训练过程是固定五步：

| 步骤 | 代码 | 含义 |
| --- | --- | --- |
| 1 | `optimizer.zero_grad()` | 清空上一轮保存的梯度 |
| 2 | `outputs = model(images)` | 前向传播，得到模型预测分数 |
| 3 | `loss = criterion(outputs, labels)` | 计算预测和真实标签之间的损失 |
| 4 | `loss.backward()` | 反向传播，计算梯度 |
| 5 | `optimizer.step()` | 更新模型参数 |

课堂重点：

- `images` 是输入图片
- `labels` 是真实类别编号
- `outputs` 是模型对每个类别的预测分数
- `loss` 越小，说明模型预测越接近真实答案

### 6.10 训练日志

```python
if batch % log_interval == 0:
    print(
        f'[train] [{epoch:03d}/{num_epochs:03d}] '
        f'[{batch:04d}/{train_dataloader_size:04d}] '
        f'loss: {loss.item():.5f}'
    )
```

这段代码每隔一定 batch 打印一次训练损失。

例如可能输出：

```text
[train] [003/050] [0010/0125] loss: 0.84231
```

含义：

- 当前是第 3 轮训练
- 一共训练 50 轮
- 当前是这一轮的第 10 个 batch
- 训练损失是 0.84231

通常情况下，随着训练进行，`loss` 应该整体下降。

### 6.11 验证模型

训练完一轮后，代码会在验证集上测试模型效果：

```python
model.eval()

with torch.no_grad():
    accuracy = 0.0

    for batch, (images, labels) in enumerate(valid_dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy /= valid_dataset_size
```

重点解释：

```python
model.eval()
```

表示切换到评估模式。

```python
with torch.no_grad():
```

表示不计算梯度，因为验证阶段只看模型效果，不更新模型参数。

```python
torch.argmax(outputs, dim=1)
```

表示从模型输出的 10 个类别分数中，取分数最高的类别作为预测结果。

```python
(torch.argmax(outputs, dim=1) == labels).sum().item()
```

表示统计这个 batch 中有多少张图片预测正确。

最后：

```python
accuracy /= valid_dataset_size
```

表示：

```text
验证准确率 = 预测正确的图片数 / 验证集图片总数
```

### 6.12 保存模型

```python
if accuracy > best_accuracy:
    best_accuracy = accuracy
    torch.save(model.state_dict(), best_checkpoint_path)

last_accuracy = accuracy
torch.save(model.state_dict(), last_checkpoint_path)
```

这里保存了两种模型：

| 文件 | 含义 |
| --- | --- |
| `best-ckpt2.pt` | 验证集准确率最高的一次模型 |
| `last-ckpt2.pt` | 最后一轮训练结束后的模型 |

为什么要保存 `best`？

因为最后一轮模型不一定是最好的。训练时间太长时，模型可能在训练集上越来越好，但在新图片上的效果反而下降，这种现象叫过拟合。

## 7. eval.py 评估流程

`eval.py` 用于评估训练好的模型在测试集上的表现。

运行命令：

```shell
python eval.py
```

评估流程：

```text
1. 读取配置
2. 定义测试集预处理
3. 加载 datasets/test
4. 创建模型结构
5. 加载训练好的 checkpoint
6. 对测试图片逐批推理
7. 统计 top1、top2、top3 准确率
8. 打印最终结果
```

### 7.1 测试集预处理

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

测试集和验证集一样，不做随机增强。

原因是测试集用于最终评价模型真实能力，必须保证输入稳定。

### 7.2 创建测试集 DataLoader

```python
dataset = ImageFolder('datasets/test', transform=transform)
dataset_size = len(dataset)

dataloader = DataLoader(
    dataset,
    batch_size=configs['batch-size'],
    num_workers=configs['num-workers'],
    shuffle=False
)
```

这里 `shuffle=False`，表示测试集不打乱顺序。

测试时是否打乱顺序通常不影响准确率，但不打乱更方便复现实验和排查问题。

### 7.3 加载模型

```python
model = FlowerNet(num_classes=configs['num-classes'], pretrained=False)
model = model.to(device)

model.load_state_dict(
    torch.load(configs['load-checkpoint-path'], map_location=device, weights_only=True)
)
model.eval()
```

这里要注意：

- `FlowerNet(...)` 创建的是模型结构
- `load_state_dict(...)` 加载的是训练好的参数
- `model.eval()` 切换到评估模式

模型结构和参数必须匹配。

如果训练时 `num-classes = 10`，评估时也必须是 10，否则最后一层参数形状会对不上。

### 7.4 top-k 准确率

`eval.py` 中计算了三个指标：

```python
top1_accuracy = 0.0
top2_accuracy = 0.0
top3_accuracy = 0.0
```

含义：

| 指标 | 含义 |
| --- | --- |
| top1 accuracy | 模型第一名预测是否正确 |
| top2 accuracy | 真实类别是否出现在模型最可能的前 2 个类别中 |
| top3 accuracy | 真实类别是否出现在模型最可能的前 3 个类别中 |

举例：

假设一张图片真实类别是 `Rose`，模型给出的前三名预测是：

```text
1. Tulip
2. Rose
3. Carnation
```

那么：

- top1 错误，因为第一名不是 `Rose`
- top2 正确，因为 `Rose` 出现在前两名中
- top3 正确，因为 `Rose` 出现在前三名中

### 7.5 torch.topk 代码解释

```python
_, top1_indices = torch.topk(outputs, 1, dim=1)
_, top2_indices = torch.topk(outputs, 2, dim=1)
_, top3_indices = torch.topk(outputs, 3, dim=1)
```

`outputs` 的形状可以理解为：

```text
[batch_size, num_classes]
```

如果 `batch_size = 32`，`num_classes = 10`，那么 `outputs` 就是：

```text
[32, 10]
```

每一行对应一张图片，每一列对应一个类别分数。

`torch.topk(outputs, 3, dim=1)` 的意思是：

```text
对每一张图片，从 10 个类别分数中取最高的 3 个类别
```

### 7.6 标签形状调整

```python
labels = labels.view(-1, 1)
```

原来的 `labels` 形状是：

```text
[batch_size]
```

例如：

```text
[3, 8, 1, 0, ...]
```

调整后形状变成：

```text
[batch_size, 1]
```

这样就可以和 `top1_indices`、`top2_indices`、`top3_indices` 做比较。

### 7.7 统计准确率

```python
top1_accuracy += (top1_indices == labels).sum().item()
top2_accuracy += (top2_indices == labels).sum().item()
top3_accuracy += (top3_indices == labels).sum().item()
```

这几行代码统计每个 batch 中预测正确的图片数量。

最后：

```python
top1_accuracy /= dataset_size
top2_accuracy /= dataset_size
top3_accuracy /= dataset_size
```

得到整个测试集上的平均准确率。

## 8. 训练、验证、测试三者区别

这是讲深度学习项目时非常重要的一点。

| 数据集 | 代码路径 | 用途 | 是否更新模型 |
| --- | --- | --- | --- |
| 训练集 | `datasets/train` | 让模型学习 | 是 |
| 验证集 | `datasets/valid` | 训练过程中挑选最好模型 | 否 |
| 测试集 | `datasets/test` | 训练完成后最终评估 | 否 |

可以这样解释：

```text
训练集：平时做练习题
验证集：阶段性模拟考试，用来判断哪个模型更好
测试集：最后正式考试，用来报告最终成绩
```

验证集和测试集不能混用。

如果一直根据测试集结果调整模型，测试集就不再是客观的最终评价。

## 9. 一张图片在项目中的完整旅程

以训练阶段为例，一张图片会经历：

```text
磁盘中的 jpg/png 文件
    ↓ ImageFolder 读取
PIL 图片
    ↓ transform 预处理和增强
Tensor: [3, 224, 224]
    ↓ DataLoader 组成 batch
Batch Tensor: [batch_size, 3, 224, 224]
    ↓ FlowerNet
模型输出: [batch_size, 10]
    ↓ CrossEntropyLoss
计算损失
    ↓ backward + optimizer.step
更新模型参数
```

以评估阶段为例：

```text
测试图片
    ↓ transform
Tensor
    ↓ FlowerNet
10 个类别分数
    ↓ topk
前 1 / 前 2 / 前 3 个预测类别
    ↓ 与真实标签比较
计算 top1 / top2 / top3 accuracy
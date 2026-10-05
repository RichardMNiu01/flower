import torch
import toml

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import FlowerNet

if __name__ == '__main__':
    # 读取训练配置，例如训练轮数、batch size、学习率、模型保存路径等。
    configs = toml.load('configs/config.toml')

    # 训练集的数据预处理和数据增强。
    # 数据增强可以人为制造更多“变化后的图片”，帮助模型提升泛化能力。
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                          # 将图片统一缩放到模型需要的输入大小。
        transforms.ColorJitter(0.1, 0.1, 0.1),                  # 随机调整亮度、对比度、饱和度。
        transforms.RandomRotation(10),                          # 随机旋转图片，增强模型对角度变化的适应性。
        transforms.RandomHorizontalFlip(),                      # 随机水平翻转图片。
        transforms.ToTensor(),                                  # 将 PIL 图片转换成 PyTorch 张量。
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 对 RGB 三个通道做标准化，加快模型收敛。
        transforms.RandomErasing(),                             # 随机擦除图片中的一小块区域，提高模型抗遮挡能力。
    ])

    # 验证集只做必要的尺寸调整和标准化，不做随机增强，保证评估结果稳定。
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    num_epochs = configs['num-epochs']

    # ImageFolder 会按照文件夹名称自动生成类别标签。
    # 例如 datasets/train/rose 下的图片会被标记为 rose 对应的类别编号。
    train_dataset = ImageFolder('datasets/train', transform=train_transform)
    valid_dataset = ImageFolder('datasets/valid', transform=valid_transform)

    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)

    # DataLoader 负责按 batch 读取数据。
    # 训练集 shuffle=True 可以打乱样本顺序，减少模型记住数据顺序的风险。
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=True)

    train_dataloader_size = len(train_dataloader)
    valid_dataloader_size = len(valid_dataloader)

    log_interval = configs['log-interval']

    best_accuracy = 0.0
    last_accuracy = 0.0

    # 根据配置选择运行设备，例如 cpu 或 cuda。
    device = torch.device(configs['device'])

    # 创建花卉分类模型，并移动到指定设备上。
    model = FlowerNet(num_classes=configs['num-classes'], pretrained=configs['load-pretrained'])
    model = model.to(device)

    # CrossEntropyLoss 常用于多分类任务；Adam 是一种常用的梯度优化器。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs['learning-rate'], weight_decay=configs['weight-decay'])

    load_checkpoint_path = configs['load-checkpoint-path']
    best_checkpoint_path = configs['best-checkpoint-path']
    last_checkpoint_path = configs['last-checkpoint-path']

    # 如果配置中要求继续训练，则先加载已有模型权重。
    if configs['load-checkpoint']:
        model.load_state_dict(torch.load(load_checkpoint_path, map_location=device, weights_only=True))

    print(f'\n---------- training start at: {device} ----------\n')

    # 外层循环控制训练轮数：每一轮都会完整遍历一次训练集。
    for epoch in range(num_epochs):
        # 切换到训练模式，启用 Dropout、BatchNorm 等训练行为。
        model.train()

        for batch, (images, labels) in enumerate(train_dataloader, start=1):
            # 将图片和标签移动到同一个设备上，保证后续计算不会报设备不一致。
            images = images.to(device)
            labels = labels.to(device)

            # 标准训练五步：清空梯度 -> 前向传播 -> 计算损失 -> 反向传播 -> 更新参数。
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch % log_interval == 0:
                print(f'[train] [{epoch:03d}/{num_epochs:03d}] [{batch:04d}/{train_dataloader_size:04d}] loss: {loss.item():.5f}')

        # 切换到评估模式，关闭 Dropout、固定 BatchNorm 统计量。
        model.eval()

        # 验证阶段不需要计算梯度，可以节省显存并加快速度。
        with torch.no_grad():
            accuracy = 0.0

            for batch, (images, labels) in enumerate(valid_dataloader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                # outputs 中每一行是一个样本在各类别上的得分。
                # argmax 取最高分对应的类别，与真实标签比较后统计预测正确的数量。
                accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

                if batch % log_interval == 0:
                    print(f'[valid] [{epoch:03d}/{num_epochs:03d}] [{batch:04d}/{valid_dataloader_size:04d}]')

            # 正确预测数除以验证集总样本数，得到本轮验证准确率。
            accuracy /= valid_dataset_size

            # 如果当前准确率超过历史最好结果，就保存为 best 模型。
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_checkpoint_path)

            # 无论效果是否最好，都保存本轮训练结束后的 last 模型。
            last_accuracy = accuracy
            torch.save(model.state_dict(), last_checkpoint_path)

        print(f'[valid] [{epoch:03d}/{num_epochs:03d}] accuracy: {accuracy:.4f}')

    print(f'best accuracy: {best_accuracy:.3f}')
    print(f'last accuracy: {last_accuracy:.3f}')

    print('\n---------- training finished ----------\n')

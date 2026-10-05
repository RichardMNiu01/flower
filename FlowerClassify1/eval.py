import torch
import toml
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import FlowerNet

if __name__ == '__main__':
    # 读取配置文件，获取 batch size、设备、类别数和模型权重路径等参数。
    configs = toml.load('configs/config.toml')

    # 测试阶段只做固定的预处理，不做随机数据增强，保证每次评估结果一致。
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                          # 将图片尺寸统一到模型输入大小。
        transforms.ToTensor(),                                  # 将图片转换为 PyTorch 张量。
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 使用和训练阶段一致的标准化方式。
    ])

    # ImageFolder 根据子文件夹名称识别类别，并为每张测试图片生成对应标签。
    dataset = ImageFolder('datasets/test', transform=transform)
    dataset_size = len(dataset)

    # 测试集不需要打乱顺序，因此 shuffle=False，便于复现实验结果。
    dataloader = DataLoader(dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=False)
    dataloader_size = len(dataloader)

    # 根据配置选择运行设备，例如 cpu 或 cuda。
    device = torch.device(configs['device'])

    # 创建模型结构。这里 pretrained=False，因为后面会加载自己训练好的权重。
    model = FlowerNet(num_classes=configs['num-classes'], pretrained=False)
    model = model.to(device)

    log_interval = configs['log-interval']

    print(f'\n---------- evaluation start at: {device} ----------\n')

    # 评估阶段不需要计算梯度，可以节省显存并加快推理速度。
    with torch.no_grad():
        top1_accuracy = 0.0
        top2_accuracy = 0.0
        top3_accuracy = 0.0

        # 加载训练阶段保存的模型参数，并切换到评估模式。
        model.load_state_dict(torch.load(configs['load-checkpoint-path'], map_location=device, weights_only=True))
        model.eval()

        for batch, (images, labels) in enumerate(dataloader, start=1):
            # 图片和标签都要移动到与模型相同的设备上。
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # topk 表示取模型认为最可能的前 k 个类别。
            # top1 是第一名预测，top2 是前两名预测，top3 是前三名预测。
            _, top1_indices = torch.topk(outputs, 1, dim=1)
            _, top2_indices = torch.topk(outputs, 2, dim=1)
            _, top3_indices = torch.topk(outputs, 3, dim=1)

            # 将标签形状从 [batch_size] 变为 [batch_size, 1]，方便和 top-k 结果逐行比较。
            labels = labels.view(-1, 1)

            # 如果真实标签出现在前 k 个预测结果中，就计为 top-k 预测正确。
            top1_accuracy += (top1_indices == labels).sum().item()
            top2_accuracy += (top2_indices == labels).sum().item()
            top3_accuracy += (top3_indices == labels).sum().item()

            if batch % log_interval == 0:
                print(f'[valid] [{batch:04d}/{dataloader_size:04d}]')

        # 正确数量除以测试集总样本数，得到最终准确率。
        top1_accuracy /= dataset_size
        top2_accuracy /= dataset_size
        top3_accuracy /= dataset_size

    print('\n--------------------------------------')
    print(f'top1 accuracy: {top1_accuracy:.3f}')
    print(f'top2 accuracy: {top2_accuracy:.3f}')
    print(f'top3 accuracy: {top3_accuracy:.3f}')

    print('\n---------- evaluation finished ----------\n')

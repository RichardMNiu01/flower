# 导入 PyTorch 中的神经网络模块
# nn 中包含卷积层、全连接层、池化层等常用神经网络组件
import torch.nn as nn

# 导入 torchvision 中已经实现好的经典网络模型
# 例如 ResNet、VGG、AlexNet 等
import torchvision.models as models


# 定义一个用于花卉分类的神经网络 FlowerNet
# 继承 nn.Module，这是所有 PyTorch 神经网络模型的基类
class FlowerNet(nn.Module):

    # 初始化函数
    # num_classes：最终需要分类的类别数量，这里默认是 10 类
    # pretrained：是否使用在 ImageNet 数据集上预训练好的 ResNet18 参数
    def __init__(self, num_classes=10, pretrained=False):

        # 调用父类 nn.Module 的初始化函数
        # 自定义 PyTorch 网络时通常都需要写这一句
        super().__init__()

        # 如果 pretrained=True
        # 就加载在 ImageNet 数据集上训练好的 ResNet18 参数
        if pretrained:

            # 创建一个 ResNet18 模型
            # weights 指定使用 ImageNet 预训练权重
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )

        # 如果 pretrained=False
        # 就不使用预训练权重，从随机初始化开始训练
        else:

            # 创建一个 ResNet18 模型
            resnet = models.resnet18()


        # =========================
        # ResNet18 第一部分：输入处理
        # =========================

        # 取出 ResNet18 的第一个卷积层
        # 原始 ResNet18 中：
        # Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3
        # )
        #
        # 如果输入：
        # [batch_size, 3, 224, 224]
        #
        # 经过 conv1 后：
        # [batch_size, 64, 112, 112]
        self.conv1 = resnet.conv1

        # 取出第一个批归一化层 BatchNorm
        # BatchNorm 可以帮助网络训练更加稳定
        #
        # 输入：
        # [batch_size, 64, 112, 112]
        #
        # 输出尺寸不变：
        # [batch_size, 64, 112, 112]
        self.bn1 = resnet.bn1

        # 取出 ReLU 激活函数
        # ReLU(x) = max(0, x)
        # 给神经网络加入非线性能力
        #
        # 输入输出尺寸不变
        self.relu = resnet.relu

        # 取出最大池化层 MaxPool
        # ResNet18 中默认：
        # kernel_size=3
        # stride=2
        # padding=1
        #
        # 输入：
        # [batch_size, 64, 112, 112]
        #
        # 输出：
        # [batch_size, 64, 56, 56]
        self.maxpool = resnet.maxpool


        # =========================
        # ResNet18 主体部分
        # =========================

        # ResNet18 的第 1 组残差网络
        # 包含 2 个 BasicBlock
        #
        # 输入：
        # [batch_size, 64, 56, 56]
        #
        # 输出：
        # [batch_size, 64, 56, 56]
        self.layer1 = resnet.layer1

        # ResNet18 的第 2 组残差网络
        # 通道数从 64 增加到 128
        # 特征图尺寸从 56×56 缩小到 28×28
        #
        # 输入：
        # [batch_size, 64, 56, 56]
        #
        # 输出：
        # [batch_size, 128, 28, 28]
        self.layer2 = resnet.layer2

        # ResNet18 的第 3 组残差网络
        # 通道数从 128 增加到 256
        # 特征图尺寸从 28×28 缩小到 14×14
        #
        # 输入：
        # [batch_size, 128, 28, 28]
        #
        # 输出：
        # [batch_size, 256, 14, 14]
        self.layer3 = resnet.layer3

        # ResNet18 的第 4 组残差网络
        # 通道数从 256 增加到 512
        # 特征图尺寸从 14×14 缩小到 7×7
        #
        # 输入：
        # [batch_size, 256, 14, 14]
        #
        # 输出：
        # [batch_size, 512, 7, 7]
        self.layer4 = resnet.layer4


        # =========================
        # 分类部分
        # =========================

        # 自适应平均池化
        # 无论输入特征图尺寸是多少
        # 最终都压缩成 1×1
        #
        # 例如：
        # [batch_size, 512, 7, 7]
        #
        # 变成：
        # [batch_size, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten 用于将多维张量展平成二维张量
        #
        # [batch_size, 512, 1, 1]
        #
        # 变成：
        # [batch_size, 512]
        self.flatten = nn.Flatten()

        # 定义最后一个全连接层
        #
        # 输入特征数量：
        # 512
        #
        # 输出特征数量：
        # num_classes
        #
        # 如果 num_classes=10：
        # [batch_size, 512]
        #       ↓
        # [batch_size, 10]
        #
        # 这 10 个数字分别表示模型对 10 个类别给出的分数
        self.fc = nn.Linear(
            in_features=512,
            out_features=num_classes
        )


    # forward 定义数据在神经网络中的前向传播过程
    # inputs 就是输入模型的一批图片
    def forward(self, inputs):

        # 输入图片首先经过第一个卷积层
        #
        # [B, 3, 224, 224]
        #       ↓
        # [B, 64, 112, 112]
        outputs = self.conv1(inputs)

        # 经过 Batch Normalization
        # 尺寸不发生变化
        #
        # [B, 64, 112, 112]
        outputs = self.bn1(outputs)

        # 经过 ReLU 激活函数
        # 增加网络的非线性表达能力
        #
        # 尺寸不发生变化
        outputs = self.relu(outputs)

        # 经过最大池化
        # 特征图尺寸缩小一半
        #
        # [B, 64, 112, 112]
        #       ↓
        # [B, 64, 56, 56]
        outputs = self.maxpool(outputs)


        # 经过 ResNet 的第一组残差块
        #
        # [B, 64, 56, 56]
        #       ↓
        # [B, 64, 56, 56]
        outputs = self.layer1(outputs)

        # 经过第二组残差块
        #
        # [B, 64, 56, 56]
        #       ↓
        # [B, 128, 28, 28]
        outputs = self.layer2(outputs)

        # 经过第三组残差块
        #
        # [B, 128, 28, 28]
        #       ↓
        # [B, 256, 14, 14]
        outputs = self.layer3(outputs)

        # 经过第四组残差块
        #
        # [B, 256, 14, 14]
        #       ↓
        # [B, 512, 7, 7]
        outputs = self.layer4(outputs)


        # 对每一个 7×7 的特征图做平均
        #
        # [B, 512, 7, 7]
        #       ↓
        # [B, 512, 1, 1]
        outputs = self.avgpool(outputs)

        # 将特征展平
        #
        # [B, 512, 1, 1]
        #       ↓
        # [B, 512]
        outputs = self.flatten(outputs)

        # 将 512 个高级特征送入最后的全连接层
        #
        # 如果 num_classes=10：
        #
        # [B, 512]
        #     ↓
        # [B, 10]
        #
        # 返回每张图片对各个类别的预测分数 logits
        return self.fc(outputs)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''-------------一、SE模块-----------------------------'''


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


'''-------------二、BasicBlock模块-----------------------------'''


# 左侧的 residual block 结构（18-layer、34-layer）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        # SE_Block放在BN之后，shortcut之前
        self.SE = SE_Block(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, self.expansion * outchannel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * outchannel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        SE_out = self.SE(out)
        out = out * SE_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


'''-------------三、Bottleneck模块-----------------------------'''


# 右侧的 residual block 结构（50-layer、101-layer、152-layer）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.conv3 = nn.Conv2d(outchannel, self.expansion * outchannel,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * outchannel)
        # SE_Block放在BN之后，shortcut之前
        self.SE = SE_Block(self.expansion * outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, self.expansion * outchannel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * outchannel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        SE_out = self.SE(out)
        out = out * SE_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


'''-------------四、搭建SE_ResNet结构-----------------------------'''


class SE_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SE_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)  # conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        return out


def SE_ResNet18():
    return SE_ResNet(BasicBlock, [2, 2, 2, 2])


def SE_ResNet34():
    return SE_ResNet(BasicBlock, [3, 4, 6, 3])


def SE_ResNet50():
    return SE_ResNet(Bottleneck, [3, 4, 6, 3])


def SE_ResNet101():
    return SE_ResNet(Bottleneck, [3, 4, 23, 3])


def SE_ResNet152():
    return SE_ResNet(Bottleneck, [3, 8, 36, 3])


'''
if __name__ == '__main__':
    model = SE_ResNet50()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
# test()
'''
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SE_ResNet50().to(device)
    # 打印网络结构和参数
    summary(net, (3, 224, 224))
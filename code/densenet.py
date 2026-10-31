"""
In ResNet, we see how the skip connection added as identity function from the inputs
to interact with the Conv layers. But in DenseNet, we see instead of adding skip 
connection to Conv layers, we can append or concat the output of identity function
with output of Conv layers.

In ResNet, it is little tedious to make the dimensions to match for adding the skip
connection and Conv Layers, but it is much simpler in DenseNet, as we concat the 
both the X and Conv's output.

The key idea or the reason its called DenseNet is because the next layers not only get
the input from previous layer but also preceeding layers before the previous layer. So 
the next layer becomes dense as it loaded with output from previous layers.

Check Figure 7.7.2 from https://d2l.ai/chapter_convolutional-modern/densenet.html for 
why DenseNet is Dense?

Two blocks comprise DenseNet, one is DenseBlock for concat operation and other is 
transition layer for controlling channels meaning dimensions (recall 1x1 Conv).
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# 把 backbone 換成 DenseNet，讓同一套 Bad-PFL + SBDADB evaluation 可以在不同模型架構下跑。

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate=12, depth=40, reduction=0.5, nClasses=10, bottleneck=True):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu = nn.ReLU(inplace=True)
        # 💡 [關鍵修改 1] 將 Pooling 註冊為正式網路層，確保 FBA 能抓取 1 維特徵！
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    # 💡 [關鍵修改 2] 讓 forward 支援標準的特徵展平，並移除 log_softmax
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        
        # 💡 [關鍵修改]：不再使用 F.relu 和 torch.flatten，全部呼叫 self 物件
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)            # 輸出原始 Logits 給 CrossEntropyLoss
        return out
    

    
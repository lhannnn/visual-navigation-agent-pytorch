import torch.nn as nn
import torch.nn.functional as F
import torch
from agent.resnet import resnet50
import numpy as np

class DQN(nn.Module):   #一个继承自 torch.nn.Module 的神经网络类 #这个类的目标是实现一个将图像输入（state）映射为动作值（Q-values）的网络
    def __init__(self): #初始化父类 nn.Module
        super(DQN, self).__init__() #super(DQN, self).__init__() 是标准写法，让 nn.Module 正确初始化。
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2) # conv1: 卷积层，输入是 RGB 图像（通道数 3），输出通道数是 16。	kernel_size=5: 卷积核大小 5x5。stride=2: 步长为 2，意味着图像尺寸会缩小。
        self.bn1 = nn.BatchNorm2d(16)    #bn1: 对 conv1 的输出做 Batch Normalization，加速收敛、提高稳定性。
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)  #输出层（全连接层）接收卷积层的输出展平后作为输入。448 是根据输入图像尺寸计算出的展平后的向量长度 ### 输出是 2 表示：有两个动作的 Q 值（比如向左、向右）

    def forward(self, x): #输入 x 是一个 batch 的图像张量，形状如 [batch_size, 3, H, W]。
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class SharedNetwork(nn.Module):
    def __init__(self):
        super(SharedNetwork, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(8192, 512)

        # Merge layer
        self.fc_merge = nn.Linear(1024, 512)

    def forward(self, inp):
        (x, y,) = inp
        
        x = x.view(-1)
        x = self.fc_siemense(x)  
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_siemense(y)
        y = F.relu(y, True)

        xy = torch.stack([x,y], 0).view(-1)
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        return xy

class SceneSpecificNetwork(nn.Module):
    """
    Input for this network is 512 tensor
    """
    def __init__(self, action_space_size):
        super(SceneSpecificNetwork, self).__init__()
        self.fc1 = nn.Linear(512, 512)

        # Policy layer
        self.fc2_policy = nn.Linear(512, action_space_size)

        # Value layer
        self.fc2_value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x_policy = self.fc2_policy(x)
        #x_policy = F.softmax(x_policy)

        x_value = self.fc2_value(x)[0]
        return (x_policy, x_value, )

class ActorCriticLoss(nn.Module):
    def __init__(self, entropy_beta):
        self.entropy_beta = entropy_beta
        pass

    def forward(self, policy, value, action_taken, temporary_difference, r):
        # Calculate policy entropy
        log_softmax_policy = torch.nn.functional.log_softmax(policy, dim=1)
        softmax_policy = torch.nn.functional.softmax(policy, dim=1)
        policy_entropy = softmax_policy * log_softmax_policy
        policy_entropy = -torch.sum(policy_entropy, 1)

        # Policy loss
        nllLoss = F.nll_loss(log_softmax_policy, action_taken, reduce=False)
        policy_loss = nllLoss * temporary_difference - policy_entropy * self.entropy_beta
        policy_loss = policy_loss.sum(0)

        # Value loss
        # learning rate for critic is half of actor's
        # Equivalent to 0.5 * l2 loss
        value_loss = (0.5 * 0.5) * F.mse_loss(value, r, size_average=False)
        return value_loss + policy_loss



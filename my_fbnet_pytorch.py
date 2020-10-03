import torch
import torch.nn as nn
from torch.nn.quantized.modules import FloatFunctional

__all__ = ['AlexNet', 'alexnet']


class FBNet_a(nn.Module):

  def __init__(self, classes=10):
    super(AlexNet, self).__init__()

    self.stage_0 = nn.Sequential(
      # stage 0_0 conv_k3 [16, 2, 1]
      nn.Conv2d(3, 16, kernel_size=3, stride=2),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
    )
      # stage 1
      # skip
    self.stage_2_0 = nn.Sequential(
      # stage 2_0:ir_k3 [24, 2, 1, e3]
      # pw
      nn.Conv2d(16, 48, kernel_size=1, stride=1),
      nn.BatchNorm2d(48),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=3//2),
      nn.BatchNorm2d(48),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(48, 24, kernel_size=1, stride=1),
      nn.BatchNorm2d(24),
    )
    self.stage_2_1_left = nn.Sequential(
      # stage 2_1:ir_k3 [24, 1, 1, e1]
      # dw
      nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=3//2),
      nn.BatchNorm2d(24),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(24, 24, kernel_size=1, stride=1),
      nn.BatchNorm2d(24),
    )
    self.stage_3_0 = nn.Sequential(
      # stage 3_0:ir_k5 [32, 2, 1, e6]
      # pw
      nn.Conv2d(24, 144, kernel_size=1, stride=1),
      nn.BatchNorm2d(144),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(144, 144, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(144),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(144, 32, kernel_size=1, stride=1),
      nn.BatchNorm2d(32),
    )
    self.stage_3_1_left = nn.Sequential(
      # stage 3_1:ir_k3 [32, 1, 1, e3]
      # pw
      nn.Conv2d(32, 96, kernel_size=1, stride=1),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=3//2),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(96, 32, kernel_size=1, stride=1),
      nn.BatchNorm2d(32),
    )
    self.stage_3_2_left = nn.Sequential(
      # stage 3_2:ir_k5 [32, 1, 1, e1]
      # dw
      nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(32, 32, kernel_size=1, stride=1),
      nn.BatchNorm2d(32),
    )
    self.stage_3_3_left = nn.Sequential(
      # stage 3_3:ir_k3 [32, 1, 1, e3]
      # pw
      nn.Conv2d(32, 96, kernel_size=1, stride=1),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=3//2),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(96, 32, kernel_size=1, stride=1),
      nn.BatchNorm2d(32),
    )
    self.stage_4_0 = nn.Sequential(
      # stage 4_0:ir_k5 [64, 2, 1, e6]
      # pw
      nn.Conv2d(32, 192, kernel_size=1, stride=1),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(192, 64, kernel_size=1, stride=1),
      nn.BatchNorm2d(64),
    )
    self.stage_4_1_left = nn.Sequential(
      # stage 4_1:ir_k5 [64, 1, 1, e3]
      # pw
      nn.Conv2d(64, 192, kernel_size=1, stride=1),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(192, 64, kernel_size=1, stride=1),
      nn.BatchNorm2d(64),
    )
    self.stage_4_2_left_aftershuffle = nn.Sequential(
      # stage 4_2:ir_k5_g2 [64, 1, 1, e1]
      # dw
      nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(64, 64, kernel_size=1, stride=1),
      nn.BatchNorm2d(64),
    )
    self.stage_4_3_left = nn.Sequential(
      # stage 4_3:ir_k5 [64, 1, 1, e6]
      # pw
      nn.Conv2d(64, 384, kernel_size=1, stride=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(384, 384, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(384, 64, kernel_size=1, stride=1),
      nn.BatchNorm2d(64),
    )
    self.stage_4_4 = nn.Sequential(
      # stage 4_4:ir_k3 [112, 1, 1, e6]
      # pw
      nn.Conv2d(64, 384, kernel_size=1, stride=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=3//2),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(384, 112, kernel_size=1, stride=1),
      nn.BatchNorm2d(112),
    )
    self.stage_4_5_left_aftershuffle = nn.Sequential(
      # stage 4_5:ir_k5_g2 [112, 1, 1, e1]
      # dw
      nn.Conv2d(112, 112, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(112),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(112, 112, kernel_size=1, stride=1),
      nn.BatchNorm2d(112),
    )
    self.stage_4_6_left = nn.Sequential(
      # stage 4_6:ir_k5 [112, 1, 1, e3]
      # pw
      nn.Conv2d(112, 336, kernel_size=1, stride=1),
      nn.BatchNorm2d(336),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(336, 336, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(336),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(336, 112, kernel_size=1, stride=1),
      nn.BatchNorm2d(112),
    )
    self.stage_4_7_left_aftershuffle = nn.Sequential(
      # stage 4_7:ir_k3_g2 [112, 1, 1, e1]
      # dw
      nn.Conv2d(112, 112, kernel_size=3, stride=1, padding=3//2),
      nn.BatchNorm2d(112),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(112, 112, kernel_size=1, stride=1),
      nn.BatchNorm2d(112),
    )
    self.stage_5_0 = nn.Sequential(
      # stage 5_0:ir_k5 [184, 2, 1, e6]
      # pw
      nn.Conv2d(112, 672, kernel_size=1, stride=1),
      nn.BatchNorm2d(672),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(672, 672, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(672),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(672, 184, kernel_size=1, stride=1),
      nn.BatchNorm2d(184),
    )
    self.stage_5_1_left = nn.Sequential(
      # stage 5_1:ir_k5 [184, 1, 1, e6]
      # pw
      nn.Conv2d(184, 1104, kernel_size=1, stride=1),
      nn.BatchNorm2d(1104),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(1104, 1104, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(1104),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(1104, 184, kernel_size=1, stride=1),
      nn.BatchNorm2d(184),
    )
    self.stage_5_2_left = nn.Sequential(
      # stage 5_2:ir_k5 [184, 1, 1, e3]
      # pw
      nn.Conv2d(184, 552, kernel_size=1, stride=1),
      nn.BatchNorm2d(552),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(552, 552, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(552),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(552, 184, kernel_size=1, stride=1),
      nn.BatchNorm2d(184),
    )
    self.stage_5_3_left = nn.Sequential(
      # stage 5_3:ir_k5 [184, 1, 1, e6]
      # pw
      nn.Conv2d(184, 1104, kernel_size=1, stride=1),
      nn.BatchNorm2d(1104),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(1104, 1104, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(1104),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(1104, 184, kernel_size=1, stride=1),
      nn.BatchNorm2d(184),
    )
    self.stage_5_4 = nn.Sequential(
      # stage 5_4:ir_k5 [352, 1, 1, e6]
      # pw
      nn.Conv2d(184, 1104, kernel_size=1, stride=1),
      nn.BatchNorm2d(1104),
      nn.ReLU(inplace=True),
      # dw
      nn.Conv2d(1104, 1104, kernel_size=5, stride=1, padding=5//2),
      nn.BatchNorm2d(1104),
      nn.ReLU(inplace=True),
      # pwl
      nn.Conv2d(1104, 352, kernel_size=1, stride=1),
      nn.BatchNorm2d(352),
    )
    self.stage_6 = nn.Sequential(
      # stage 6:conv_k1 [1504, 1, 1]
      nn.Conv2d(352, 1504, kernel_size=1, stride=1),
      nn.BatchNorm2d(1504),
      nn.ReLU(inplace=True),
    )
    self.head_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.head_conv = nn.Conv2d(1504, 1000, 1)


  def forward(self, x):




    x = x.view(x.size(0), -1)
    return x


def alexnet(**kwargs):
  r"""AlexNet model architecture from the
  `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
  """
  model = AlexNet(**kwargs)
  return model



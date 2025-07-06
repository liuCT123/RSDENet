# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from backbone.origin.from_origin import Backbone_VGG16_in3
from module.BaseBlocks import BasicConv2d
from module.MyModule import *
from utils.tensor_ops import cus_sample
from backbone.origin.res2net_v1b_base import Res2Net_model
from backbone.origin.swin import *
#VGG-16 backbone
class RSDENet(nn.Module):
    def __init__(self):
        super(RSDENet, self).__init__()

        self.upsample = cus_sample

        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        self.Gate1 = CBAM(64)
        self.Gate2 = CBAM(128)
        self.Gate3 = CBAM(256)
        self.Gate4 = CBAM(512)
        self.Gate5 = CBAM(512)

        #1*1 conv change channel
        self.con_AIM5 = nn.Conv2d(512, 64, 3, 1, 1)
        self.con_AIM4 = nn.Conv2d(512, 64, 3, 1, 1)
        self.con_AIM3 = nn.Conv2d(256, 64, 3, 1, 1)
        self.con_AIM2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.con_AIM1 = nn.Conv2d(64, 32, 3, 1, 1)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.convertChannels = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        # 新增反射抑制模块
        self.RSM1 = ReflectionSuppressionModule(in_channels=32)
        self.RSM2 = ReflectionSuppressionModule(in_channels=64)
        self.RSM3 = ReflectionSuppressionModule(in_channels=64)
        self.RSM4 = ReflectionSuppressionModule(in_channels=64)
        self.RSM5 = ReflectionSuppressionModule(in_channels=64)

        # 添加MIB模块
        self.mib5 = DE_DIB(64, 64)  # 第五层
        self.mib4 = DE_DIB(64, 64)  # 第四层
        self.mib3 = DE_DIB(64, 64)  # 第三层
        self.mib2 = DE_DIB(64, 64)  # 第二层
        self.mib1 = DE_DIB(32, 32)  # 第一层


        self.classifier = nn.Conv2d(32, 1, 1)
        self.classifierp6 = nn.Conv2d(64, 1, 1)
        self.predtrans = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.predtrans64 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, in_data):
        in_data_1 = self.encoder1(in_data)  # 64通道 256x256
        in_data_2 = self.encoder2(in_data_1)  # 128通道 128x128
        in_data_4 = self.encoder4(in_data_2)  # 256通道 64x64
        in_data_8 = self.encoder8(in_data_4)  # 512通道 32x32
        in_data_16 = self.encoder16(in_data_8)  # 512通道 16x16

        in_data_1 = self.Gate1(in_data_1)
        in_data_2 = self.Gate2(in_data_2)
        in_data_4 = self.Gate3(in_data_4)
        in_data_8 = self.Gate4(in_data_8)
        in_data_16 = self.Gate5(in_data_16)


        in_data_16 = self.con_AIM5(in_data_16)  # 512->64通道

        # 1x1 Conv 改变其他层通道数
        in_data_1 = self.con_AIM1(in_data_1)   # 32 
        in_data_2 = self.con_AIM2(in_data_2)   # 64
        in_data_4 = self.con_AIM3(in_data_4)   # 64
        in_data_8 = self.con_AIM4(in_data_8)   # 64

        # RSM模块处理
        in_data_1 = self.RSM1(in_data_1)       # RSM模块，32
        in_data_2 = self.RSM2(in_data_2)       # 64
        in_data_4 = self.RSM3(in_data_4)       # 64
        in_data_8 = self.RSM4(in_data_8)       # 64
        in_data_16 = self.RSM5(in_data_16)     # 64

        # 第五层特征融合
        out_data_16 = self.mib5(in_data_16, in_data_16)
        out_data_16 = self.upconv16(out_data_16)

        # 第四层特征融合
        out_data_8 = self.mib4(in_data_8, out_data_16)
        out_data_8 = self.upconv8(out_data_8)

        # 第三层特征融合
        out_data_4 = self.mib3(in_data_4, out_data_8)
        out_data_4 = self.upconv4(out_data_4)

        # 第二层特征融合
        out_data_2 = self.mib2(in_data_2, out_data_4)
        out_data_2 = self.upconv2(out_data_2)

        # 第一层特征融合
        out_data_1 = self.mib1(in_data_1, self.convertChannels(out_data_2))
        out_data_1 = self.upconv1(out_data_1)

        # 分类预测
        out_data = self.classifier(out_data_1)
        s1 = F.interpolate(self.predtrans(out_data), size=in_data.shape[2:], mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.predtrans64(out_data_2), size=in_data.shape[2:], mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.predtrans64(out_data_4), size=in_data.shape[2:], mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.predtrans64(out_data_8), size=in_data.shape[2:], mode='bilinear', align_corners=True)
        s5 = F.interpolate(self.predtrans64(out_data_16), size=in_data.shape[2:], mode='bilinear', align_corners=True)

        return s1, s2, s3, s4, s5

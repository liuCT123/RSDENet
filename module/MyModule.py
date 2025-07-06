import torch
# import torch.nn as nn
from utils.tensor_ops import cus_sample
# import torch
# import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import scipy.stats as st

def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)

import torch.nn.functional as F
import math
eps = 1e-12


class ReflectionSuppressionModule(nn.Module):
    def __init__(self, initial_threshold=30, in_channels=3, dilations=[1, 2, 3]):
        """
        Args:
          - initial_threshold: 初始的梯度阈值（会作为可学习参数）
          - in_channels: 输入图像的通道数
          - dilations: 用于多尺度梯度计算的膨胀率列表
        """
        super(ReflectionSuppressionModule, self).__init__()
        # 将阈值设为可学习参数
        self.threshold = nn.Parameter(torch.tensor(float(initial_threshold)))
        # self.threshold = initial_threshold
        self.in_channels = in_channels
        # 用于特征提取的卷积层
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        # 定义用于计算梯度的固定卷积核
        kernel_x = torch.tensor([[-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 1, 2)
        kernel_y = torch.tensor([[-1], [1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 2, 1)
        # 将每个通道都有一份
        self.register_buffer('kernel_x', kernel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer('kernel_y', kernel_y.repeat(in_channels, 1, 1, 1))
        # 保存多尺度使用的膨胀率列表
        self.dilations = dilations
        # 融合卷积：输入通道数为 in_channels * 2 * len(dilations)（水平+垂直各一份），输出与原通道数一致
        # self.fuse_conv = nn.Conv2d(in_channels * 2 * len(dilations), in_channels, kernel_size=3, padding=1)

        self.fuse_conv_seq = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1)

    def poisson_solver_fft(self, rhs, eps=1e-6):
        """
        使用 FFT 解 Poisson 方程（假设周期边界条件）

        求解: Δu = rhs
        """
        B, C, H, W = rhs.shape
        # 构造频域网格
        i = torch.arange(H, device=rhs.device, dtype=rhs.dtype).view(H, 1)
        j = torch.arange(W, device=rhs.device, dtype=rhs.dtype).view(1, W)
        lambda_x = 2 - 2 * torch.cos(2 * np.pi * i / H)
        lambda_y = 2 - 2 * torch.cos(2 * np.pi * j / W)
        eigen = lambda_x + lambda_y  # (H, W)
        eigen[0, 0] = 1.0  # 避免除零
        # FFT求解
        rhs_fft = torch.fft.fft2(rhs)
        u_fft = rhs_fft / (eigen + eps)
        u_fft[..., 0, 0] = 0.0  # 保持DC分量为0
        u = torch.fft.ifft2(u_fft).real
        return u
    def multi_scale_gradient(self, x):
        """
        逐尺度计算梯度并直接累加融合，
        不再对不同尺度的梯度进行通道拼接，从而降低显存占用。
        """
        fused_grad = 0
        for d in self.dilations:
            # 水平梯度：右侧填充 d 列
            x_pad_x = F.pad(x, (0, d, 0, 0), mode='replicate')
            grad_x = F.conv2d(x_pad_x, self.kernel_x, groups=self.in_channels, dilation=d)
            # 垂直梯度：底部填充 d 行
            x_pad_y = F.pad(x, (0, 0, 0, d), mode='replicate')
            grad_y = F.conv2d(x_pad_y, self.kernel_y, groups=self.in_channels, dilation=d)
            # 将水平和垂直梯度直接相加，得到当前尺度的梯度信息
            scale_grad = grad_x + grad_y
            # 累加各尺度的梯度
            fused_grad = fused_grad + scale_grad
        # 对所有尺度的累加结果取平均
        fused_grad = fused_grad / len(self.dilations)
        # 通过卷积层进一步融合，输出与输入通道数一致
        fused_grad = self.fuse_conv_seq(fused_grad)
        return fused_grad

    def forward(self, img):
        """
        前向传播流程：
          1. 计算多尺度梯度并融合
          2. 对融合的梯度进行阈值过滤
          3. 利用 Poisson 求解器重构图像
          4. 计算原图与重构图的差值并与原图相加，再通过卷积提取平滑特征
        """
        # 计算多尺度融合后的梯度
        fused_grad = self.multi_scale_gradient(img)

        # fused_grad = self.conv(img)

        # # 对梯度绝对值进行阈值过滤（硬阈值）
        # mask = (fused_grad.abs() > self.threshold).float()
        # grad_thresh = fused_grad * mask

        # 软阈值处理
        scale_factor = 10.0  # 控制 Sigmoid 软阈值的陡峭程度
        soft_mask = torch.sigmoid((self.threshold - fused_grad.abs()) * scale_factor) 
        grad_thresh = fused_grad * soft_mask  # 这里改成软阈值

        # 通过 Poisson 求解器重构图像
        u = self.poisson_solver_fft(-grad_thresh)

        # # 用卷積替代FFT
        # u =self.conv(grad_thresh)

        # 殘差鏈接
        output = self.conv(img + u)
        return output


# --- Difference-Enhanced Dual Interaction Block (DE-DIB) ---
class DE_DIB(nn.Module):
    def __init__(self, skip_channels, deep_channels):
        super(DE_DIB, self).__init__()
        self.out_channels = skip_channels
        self.mid_channels = skip_channels # Assuming using skip_channels as common dim

        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Feature preparation convolutions ---
        # Process concatenated features for both scales initially
        # Output channels = mid_channels (e.g., 128) for further processing
        self.conv_large_in = BasicConv2d(skip_channels + deep_channels, self.mid_channels, kernel_size=3, padding=1)
        self.conv_small_in = BasicConv2d(skip_channels + deep_channels, self.mid_channels, kernel_size=3, padding=1)

        # --- Optional: Deeper processing blocks for each scale (can use CBAM or just Conv) ---
        # Let's use a simple BasicConv2d for this example
        self.process_large = BasicConv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)
        self.process_small = BasicConv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)

        # *** NEW: Add CBAM before final fusion ***
        self.cbam_fuse_large = CBAM(self.mid_channels)
        self.cbam_fuse_small = CBAM(self.mid_channels)

        # *** NEW: Add CBAM for the difference feature ***
        self.cbam_diff = CBAM(self.mid_channels)

        # --- Difference Enhancement ---
        # Convolution to process the difference feature. Output channels = mid_channels
        self.conv_diff = BasicConv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)
        # Optional: Add activation or normalization after adding diff? ReLU might suppress negative diffs.
        # self.relu_after_add = nn.ReLU(inplace=True) # Or use Tanh? Or nothing?

        # --- Final Fusion ---
        # Choose a fusion strategy for enhanced large scale and upsampled small scale
        # Option 1: Multiplicative Fusion (like original MIB)
        # self.conv_fuse_mul = BasicConv2d(self.mid_channels, self.out_channels, kernel_size=3, padding=1)

        # Option 2: Concatenation Fusion
        # self.conv_fuse_cat = BasicConv2d(self.mid_channels * 2, self.out_channels, kernel_size=3, padding=1)

        # Option 3: Additive Fusion
        self.conv_fuse_add_align = BasicConv2d(self.mid_channels, self.mid_channels, kernel_size=1) # Align channels if needed before add
        self.conv_fuse_add_out = BasicConv2d(self.mid_channels, self.out_channels, kernel_size=3, padding=1)


    def forward(self, x, y):
        # x: feature from skip connection (higher res)
        # y: feature from previous (deeper) decoder stage (lower res)

        # --- Prepare inputs for both scales ---
        x_large = x
        if x.size()[2:] == y.size()[2:]: # Deepest stage
            y_large = y; x_small = self.down2(x); y_small = self.down2(y)
        elif x.size()[2] == y.size()[2] * 2: # Standard decoder
            y_large = self.up2(y); x_small = self.down2(x); y_small = y
        else: # Fallback
            print(f"Warning: Non-standard size relation in DE_DIB. x:{x.shape}, y:{y.shape}. Interpolating.")
            y_large = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
            y_small = y; x_small = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True)

        # --- Initial Processing ---
        feat_large_prep = self.conv_large_in(torch.cat((x_large, y_large), dim=1))
        feat_small_prep = self.conv_small_in(torch.cat((x_small, y_small), dim=1))

        # --- Optional Deeper Processing ---
        feat_large = self.process_large(feat_large_prep)
        feat_small = self.process_small(feat_small_prep)

        # --- Upsample Small Scale Feature ---
        feat_small_up = self.up2(feat_small)
        # Ensure alignment
        if feat_small_up.shape[2:] != feat_large.shape[2:]:
             feat_small_up = F.interpolate(feat_small_up, size=feat_large.shape[2:], mode='bilinear', align_corners=True)

        # --- Calculate and Process Difference ---
        difference = feat_large - feat_small_up # Calculate the high-frequency details
        processed_diff = self.conv_diff(difference)
        processed_diff = self.cbam_diff(processed_diff) # *** Apply CBAM here ***

        # --- Enhance Large Scale Feature ---
        feat_large_enhanced = feat_large + processed_diff
        # Optional activation after adding difference:
        # feat_large_enhanced = self.relu_after_add(feat_large_enhanced)

        # *** Apply CBAM before final fusion ***
        feat_large_enhanced = self.cbam_fuse_large(feat_large_enhanced)
        feat_small_up = self.cbam_fuse_small(feat_small_up)

        # --- Final Fusion ---
        # Choose ONE fusion method:

        # Method 1: Multiplicative Fusion (like MIB)
        # final_fused = torch.mul(feat_large_enhanced, feat_small_up) # Use enhanced large feature
        # out = self.conv_fuse_mul(final_fused)

        # Method 2: Concatenation Fusion
        # final_fused = torch.cat((feat_large_enhanced, feat_small_up), dim=1)
        # out = self.conv_fuse_cat(final_fused)

        # Method 3: Additive Fusion
        feat_small_up_aligned = self.conv_fuse_add_align(feat_small_up) # Align if needed
        final_fused = feat_large_enhanced + feat_small_up_aligned
        out = self.conv_fuse_add_out(final_fused)

        return out

    
    
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    
class Contrast(nn.Module):
    def __init__(self, in_c):
        super(Contrast, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1,padding=1)
        self.conv_1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = x-self.avg_pool(x)  #Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight*x + x

        return out#Res

    
#CBAM
class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channel)

    def forward(self, x):
        # CA = x.mul(self.ca(x))
        # # 元素级别点对点相乘
        # SA = CA.mul(self.sa(CA))


        x1_ca = x.mul(self.ca(x))
        x1_sa = x1_ca.mul(self.sa(x1_ca))
        x = x + x1_sa
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):   #CA
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):  #SA
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    module = CSC(64)
    # print([(name, params.size()) for name, params in module.named_parameters()])

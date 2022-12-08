import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm, in_size_up=None, out_size_up=None):
        super(unetUp, self).__init__()

        if not(is_deconv):
            in_size = int(in_size * 1.5)
        self.conv = unetConv2(in_size, out_size, is_batchnorm)

        if in_size_up is None:
            in_size_up = in_size
        if out_size_up is None:
            out_size_up = out_size

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size_up, out_size_up, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset_2 = inputs1.size(2) - outputs2.size(2)
        offset_3 = inputs1.size(3) - outputs2.size(3)
        padding = [0, offset_3, 0, offset_2]
        outputs2 = F.pad(outputs2, padding)

        return self.conv(torch.cat([inputs1, outputs2], 1))


class UNet_res18(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=25,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
        replace_stride_with_dilation=None
    ):
        super(UNet_res18, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder
        self.conv0 = nn.Sequential(
            conv3x3(in_channels, filters[0], stride=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        ds1 = nn.Sequential(
            conv1x1(filters[0], filters[1], stride=2),
            nn.BatchNorm2d(filters[1], eps=1e-05),
        )
        self.conv1 = nn.Sequential(
            BasicBlock(filters[0], filters[1], stride=2, downsample=ds1),
            BasicBlock(filters[1], filters[1], stride=1, downsample=None)
        )

        ds2 = nn.Sequential(
            conv1x1(filters[1], filters[2], stride=2),
            nn.BatchNorm2d(filters[2], eps=1e-05),
        )
        self.conv2 = nn.Sequential(
            BasicBlock(filters[1], filters[2], stride=2, downsample=ds2),
            BasicBlock(filters[2], filters[2], stride=1, downsample=None)
        )

        ds3 = nn.Sequential(
            conv1x1(filters[2], filters[3], stride=2),
            nn.BatchNorm2d(filters[3], eps=1e-05),
        )
        self.conv3 = nn.Sequential(
            BasicBlock(filters[2], filters[3], stride=2, downsample=ds3),
            BasicBlock(filters[3], filters[3], stride=1, downsample=None)
        )

        ds4 = nn.Sequential(
            conv1x1(filters[3], filters[4], stride=2),
            nn.BatchNorm2d(filters[4], eps=1e-05),
        )
        self.conv4 = nn.Sequential(
            BasicBlock(filters[3], filters[4], stride=2, downsample=ds4),
            BasicBlock(filters[4], filters[4], stride=1, downsample=None)
        )

        # decoder
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, inputs):
        c0 = self.conv0(inputs)         # 64 x H x W
        c1 = self.conv1(c0)             # 128 x H/2 x W/2
        c2 = self.conv2(c1)             # 256 x H/4 x W/4
        c3 = self.conv3(c2)             # 512 x H/8 x W/8
        c4 = self.conv4(c3)             # 1024 x H/16 x W/16
        up4 = self.up_concat4(c3, c4)   # 512 x H/8 x W/8
        up3 = self.up_concat3(c2, up4)  # 256 x H/4 x W/4
        up2 = self.up_concat2(c1, up3)  # 128 x H/2 x W/2
        up1 = self.up_concat1(c0, up2)  # 64 x H x W

        final = self.final(up1)         # C x H x W

        return final

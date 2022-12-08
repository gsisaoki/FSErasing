import torch
from torch import nn
from classifier import ArcFace

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     groups=groups,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, use_bn1=True, group_norm=False):
        super(IBasicBlock, self).__init__()
        #if groups != 1 or base_width != 64:
        #    raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.use_bn1 = use_bn1
        if use_bn1:
            if not(group_norm):
                self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
            else:
                self.bn1 = nn.GroupNorm(groups, inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes, groups=groups)

        if not(group_norm):
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        else:
            self.bn2 = nn.GroupNorm(groups, planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups=groups)

        if not(group_norm):
            self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        else:
            self.bn3 = nn.GroupNorm(groups, planes, eps=1e-05,)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.use_bn1:
            out = self.bn1(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False,
                 arcface=False, num_classes=10572, s=64, m=0.5, input_channels=3, group_norm=False, ch_scale=1.0):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = int(64 * ch_scale)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.group_norm = group_norm

        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, int(64 * ch_scale), layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       int(128 * ch_scale),
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       int(256 * ch_scale),
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       int(512 * ch_scale),
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(int(512 * ch_scale) * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(int(512 * ch_scale) * block.expansion * self.fc_scale, num_features)

        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        if arcface:
            self.classifier = ArcFace(num_features, num_classes, margin_arc=m, scale=s)
        else:
            self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not(self.group_norm):
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, groups=self.groups),
                    nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, groups=self.groups),
                    nn.GroupNorm(self.groups, planes * block.expansion, eps=1e-05, ),
                )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, group_norm=self.group_norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      group_norm=self.group_norm))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)

        ft = x
        if labels is None:
            x = self.classifier(ft)
        else:
            x = self.classifier(ft, labels)
        return x, ft



def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

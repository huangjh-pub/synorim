import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class BasicBlockBase(nn.Module):
    """
    A double-conv ResBlock with relu activation, with residual connection.
    """
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, D=3):
        super(BasicBlockBase, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D)
        self.norm1 = ME.MinkowskiInstanceNorm(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, dimension=D)
        self.norm2 = ME.MinkowskiInstanceNorm(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class ResUNet(ME.MinkowskiNetwork):
    """
    Our main network structure - a U-Net with residual double-conv blocks.
        Please refer to the appendix of our paper for illustration of the model.
    """
    def __init__(self, network_config,
                 in_channels=3, out_channels=32, normalize_feature=None, conv1_kernel_size=None, D=3):
        super().__init__(D)
        channels = network_config.channels
        tr_channels = list(network_config.tr_channels)

        assert len(channels) == len(tr_channels)
        channels[0] = in_channels
        tr_channels.append(0)

        self.normalize_feature = normalize_feature

        self.in_convs, self.in_norms, self.in_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.out_convs, self.out_norms, self.out_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for layer_id in range(len(channels) - 1):
            self.in_convs.append(ME.MinkowskiConvolution(
                in_channels=channels[layer_id],
                out_channels=channels[layer_id + 1],
                kernel_size=conv1_kernel_size if layer_id == 0 else 3,
                stride=1 if layer_id == 0 else 2,
                dilation=1, bias=False, dimension=D))
            self.in_norms.append(ME.MinkowskiInstanceNorm(channels[layer_id + 1]))
            self.in_blocks.append(BasicBlockBase(
                                      channels[layer_id + 1], channels[layer_id + 1], D=D))
            self.out_convs.append(ME.MinkowskiConvolutionTranspose(
                in_channels=channels[layer_id + 1] + tr_channels[layer_id + 2],
                out_channels=tr_channels[layer_id + 1],
                kernel_size=1 if layer_id == 0 else 3,
                stride=1 if layer_id == 0 else 2,
                dilation=1,
                bias=False,
                dimension=D))
            if layer_id > 0:
                self.out_norms.append(ME.MinkowskiInstanceNorm(tr_channels[layer_id + 1]))
                self.out_blocks.append(BasicBlockBase(
                    tr_channels[layer_id + 1], tr_channels[layer_id + 1], D=D))

        self.final = ME.MinkowskiConvolution(
            in_channels=tr_channels[1], out_channels=out_channels,
            kernel_size=1, stride=1, dilation=1, bias=True, dimension=D)

    def forward(self, x):
        skip_outputs = []
        for layer_id in range(len(self.in_convs)):
            out_skip = self.in_convs[layer_id](x)
            out_skip = self.in_norms[layer_id](out_skip)
            out_skip = self.in_blocks[layer_id](out_skip)
            x = MEF.relu(out_skip)
            skip_outputs.append(out_skip)

        for layer_id in range(len(self.in_convs) - 1, -1, -1):
            x = self.out_convs[layer_id](x)
            if layer_id > 0:
                x = self.out_norms[layer_id - 1](x)
                x = self.out_blocks[layer_id - 1](x)
            x_tr = MEF.relu(x)
            if layer_id > 0:
                x = ME.cat(x_tr, skip_outputs[layer_id - 1])

        out = self.final(x)

        if self.normalize_feature:
            return ME.SparseTensor(
                out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager)
        else:
            return out

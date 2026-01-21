import torch
import torch.nn as nn


class KAMConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, poly_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=None, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0,
                 **norm_kwargs):
        super(KAMConvNDLayer, self).__init__()
        if poly_order < 0:
            raise ValueError('poly_order must be a non-negative integer')
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.inputdim = input_dim
        self.outdim = output_dim
        self.poly_order = poly_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.norm_kwargs = norm_kwargs

        grid_min, grid_max = float(grid_range[0]), float(grid_range[1])
        if grid_max == grid_min:
            raise ValueError('grid_range must have different min and max values')
        self.register_buffer('range_center', torch.tensor((grid_min + grid_max) / 2.0))
        self.register_buffer('range_scale', torch.tensor((grid_max - grid_min) / 2.0))

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        cheb_in_dim = (poly_order + 1) * input_dim // groups
        self.cheb_conv = nn.ModuleList([conv_class(cheb_in_dim,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.cheb_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def _chebyshev_bases(self, x):
        x_scaled = (x - self.range_center) / (self.range_scale + 1e-6)
        x_scaled = torch.tanh(x_scaled)

        t0 = torch.ones_like(x_scaled)
        if self.poly_order == 0:
            return t0.unsqueeze(2)

        t1 = x_scaled
        bases = [t0, t1]
        for _ in range(2, self.poly_order + 1):
            bases.append(2 * x_scaled * bases[-1] - bases[-2])
        return torch.stack(bases, dim=2)

    def forward_kam(self, x, group_index):
        base_output = self.base_conv[group_index](self.base_activation(x))

        bases = self._chebyshev_bases(x)
        bases = bases.flatten(1, 2)
        cheb_output = self.cheb_conv[group_index](bases)

        x = self.prelus[group_index](self.layer_norm[group_index](base_output + cheb_output))

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kam(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KAMConv3DLayer(KAMConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, poly_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=None, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0, norm_layer=nn.InstanceNorm3d,
                 **norm_kwargs):
        super(KAMConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                             input_dim, output_dim,
                                             poly_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=3,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KAMConv2DLayer(KAMConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, poly_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=None, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super(KAMConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                             input_dim, output_dim,
                                             poly_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=2,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KAMConv1DLayer(KAMConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, poly_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=None, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0, norm_layer=nn.InstanceNorm1d,
                 **norm_kwargs):
        super(KAMConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                             input_dim, output_dim,
                                             poly_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=1,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)

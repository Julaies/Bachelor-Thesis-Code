import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class CustomModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = None
        self.R = None

    def relprop(self, R):
        self.R = R
        return R


# Define new layer including relevance propagation
class CustomReLU(CustomModule, nn.ReLU):
    def forward(self, input):
        self.x = input
        return super().forward(input)

class CustomLeakyReLU(CustomModule, nn.LeakyReLU):
    def forward(self, input):
        self.x = input
        return super().forward(input)

class CustomLinear(CustomModule, nn.Linear):
    def forward(self, input):
        #restrict bias to negative values 
        self.bias.data = torch.clamp(self.bias, max=0) - torch.clamp(self.bias, min=0) * 1e-4
        self.x = input
        return F.linear(input, self.weight, self.bias)


class NextLinear(CustomLinear):
    def relprop(self, R):
        weights = self.weight.cpu().detach().numpy()
        activation = self.x.cpu().detach().numpy()
        V = np.maximum(0, weights)
        Z = np.dot(activation, V.T)+1e-9
        S = R/Z
        C = np.dot(S, V)
        R = activation*C
        self.R = R
        return R

class CustomConv3d(CustomModule, nn.Conv3d):
    def forward(self, input):
        #self.bias.data = torch.clamp(self.bias, max=0) - torch.clamp(self.bias, min=0) * 1e-4
        self.x = input
        return super().forward(input)


class FirstConv3d(CustomConv3d):
    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)
        #implement w^2 propagation rule 
        weights = copy.deepcopy(self.weight)
        weights = weights**2 
        weights = weights * 1/torch.sum(weights, dim=(-1,-2,-3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #print('weights', weights)

        iself = FirstConv3d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        iself.bias.data *= 0
        iself.weight.data = weights
        
        x = torch.ones_like(self.x, requires_grad=True)
        z = iself.forward(x)
        shape = z.shape
        R = R.view(shape)
        z.backward(R+1e-9)
        C = x.grad
        
        self.R = C
        return C

class NextConv3d(CustomConv3d):
    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)

        pself = CustomConv3d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        pself.bias.data *= 0
        pself.weight.data = torch.clamp_min(copy.deepcopy(self.weight), 0)

        x = copy.deepcopy(self.x.data).requires_grad_()
        Z = pself.forward(x)
        shape = Z.shape

        R = R.view(shape)
        S = R.squeeze().clone().detach() / (Z.squeeze() + 1e-9)

        Z.backward(S.view(shape))
        C = x.grad
        R = self.x*C
        self.R = R
        return R


class CustomMaxPool3d(CustomModule, nn.MaxPool3d):
    def forward(self, input):
        self.x = input
        return super().forward(input)

    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)
        x = copy.deepcopy(self.x.data).requires_grad_()
        Z = self.forward(x)
        shape = Z.shape
        R = R.view(shape)
        S = R.squeeze().clone().detach() / (Z.squeeze() + 1e-9)
        # R.squeeze().clone().detach()
        Z.backward(S.view(shape))
        C = x.grad

        self.R = x * C
        return self.R

class CustomSumPool3d(CustomModule, nn.AvgPool3d):
    def forward(self, input):
        self.x = input
        return super().forward(input*16**3)

    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)
        x = copy.deepcopy(self.x.data).requires_grad_()
        Z = self.forward(x)
        shape = Z.shape
        R = R.view(shape)
        S = R.squeeze().clone().detach() / (Z.squeeze() + 1e-9)
        # R.squeeze().clone().detach()
        Z.backward(S.view(shape))
        C = x.grad

        self.R = x * C
        return self.R


class _CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []

    def relprop(self, relevance, layer_stop = 0):
        for layer in self.layers[::-1]:
            relevance = layer.relprop(relevance)
            if layer == self.layers[layer_stop]: break  
        return relevance


class _Conv3dBlock(_CustomModel):
    def __init__(self, n_filters):
        super().__init__()
        self.conv = nn.Sequential(
            FirstConv3d(1, n_filters, 2, bias=False),
            CustomReLU(),
            CustomMaxPool3d(kernel_size = 2)
        )
        self.init_layers()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        N = input.shape[-1]         # lattice size
        #print(self.conv(input.view(batch_size, 1, N, N, N)).shape)
        return self.conv(input.view(batch_size, 1, N, N, N))

    def init_layers(self) -> None:
        for layer in self.conv.children():
            self.layers.append(layer)


class CNN_Classification(_CustomModel):
    def __init__(self, out_shape, n_filters, n_dense_nodes):
        super().__init__()
        self.out_shape = out_shape
        assert isinstance(out_shape, np.ndarray), 'type(out_shape) is not np.ndarray'
        
        self.conv = _Conv3dBlock(n_filters)
        self.dense = nn.Sequential(
            NextLinear(n_dense_nodes, 512),
            CustomReLU(),
            NextLinear(512, self.out_shape.prod()),
            CustomLeakyReLU(),
        )
        self.init_layers()
        #self.index_conv_layer = self._get_index_conv_layers()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert isinstance(input, torch.Tensor), f'input hat incorrect type: {type(input)}'
        batch_size = input.shape[0]
        x = F.pad(input, (0,1,0,1,0,1), mode = "circular")
        x = self.conv(x)
        x = x.view(batch_size, -1)
        #print(x.shape)
        x = self.dense(x)
        shape = tuple(np.insert(self.out_shape, 0, batch_size))
        return x.view(shape)

    def init_layers(self) -> None:
        self.layers = self.conv.layers
        for layer in self.dense.children():
            self.layers.append(layer)
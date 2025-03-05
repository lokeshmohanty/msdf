from ctypes import Union
from flax import nnx
from typing import Callable
from functools import partial
import jax.numpy as jnp
import jax

rngs = nnx.Rngs(0)

class ResidualBlock(nnx.Module):
    scale: int = 1
    layers: list = []
    downsample: Callable | None

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        for layer in self.layers:
            x = layer(x)

        if self.downsample: residual = self.downsample(residual)
        x += residual
        return nnx.relu(x)

class BasicBlock(ResidualBlock):
    layers: list = []
    downsample: Callable | None

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride = (1, 1), 
                 downsample: Callable | None = None):
        self.downsample = downsample
        self.layers = [
            nnx.Conv(in_channels, out_channels, (3, 3), strides=stride, use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, use_running_average=False, rngs=rngs),
            nnx.relu,

            nnx.Conv(out_channels, out_channels, (3, 3), strides=stride, use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, use_running_average=False, rngs=rngs),
        ]

class Bottleneck(ResidualBlock):
    scale: int = 4
    layers: list = []
    downsample: Callable | None

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride = (1, 1), 
                 downsample: Callable | None = None):
        self.downsample = downsample
        self.layers = [
            nnx.Conv(in_channels, out_channels, (1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, use_running_average=False, rngs=rngs),
            nnx.relu,

            nnx.Conv(out_channels, out_channels, (3, 3), strides=stride, use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, use_running_average=False, rngs=rngs),
            nnx.relu,

            nnx.Conv(out_channels, out_channels * self.scale, (1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels * self.scale, use_running_average=False, rngs=rngs),
        ]


class Resnet(nnx.Module):
    block: ResidualBlock

    def __init__(self, 
                 block: ResidualBlock, 
                 layers: list[int], 
                 num_output: int, 
                 *, rngs: nnx.Rngs):
        self.in_channels = 64
        self.block = block
        self.layers = [
            nnx.Conv(3, 64, (7, 7), strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(64, rngs=rngs),
            nnx.relu,
            partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),

            *self.make_layers(64, layers[0]),
            *self.make_layers(128, layers[1], stride=(2, 2)),
            *self.make_layers(256, layers[2], stride=(2, 2)),
            *self.make_layers(512, layers[3], stride=(2, 2)),

            *self.make_deconv_layers(3, [256] * 3, [(4, 4)] * 3),

            nnx.Conv(256, 64, (3, 3), rngs=rngs),
            nnx.relu,
            nnx.Conv(64, num_output, (1, 1), rngs=rngs),

            lambda x: jnp.transpose(x, (0, 3, 1, 2)),
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x

    def make_layers(
        self, 
        out_channels: int, 
        num_layers: int, 
        stride = (1, 1)
    ) -> list:
        downsample = None
        if stride != (1, 1) or self.in_channels != out_channels * self.block.scale:
            downsample = nnx.Sequential(
                nnx.Conv(self.in_channels, out_channels * self.block.scale, (1, 1), stride, use_bias=False, rngs=rngs),
                nnx.BatchNorm(out_channels * self.block.scale, rngs=rngs),
            )
        layers = [self.block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * self.block.scale
        layers += [self.block(self.in_channels, out_channels)] * (num_layers - 1)
        return layers
   
    def make_deconv_layers(
        self, 
        num_layers: int, 
        out_channels: list, 
        kernels: list
    ) -> list:
        layers: list = []
        for i in range(num_layers):
            layers += [
                nnx.ConvTranspose(self.in_channels, out_channels[i], kernels[i], (2, 2), rngs=rngs),
                nnx.BatchNorm(out_channels[i], rngs=rngs),
                nnx.relu,
            ]
            self.in_channels = out_channels[i]
        return layers

def make_resnet(size: int):
    spec = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
    }
    return partial(Resnet, *spec[size])

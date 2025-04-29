# Alternative implementations: https://github.com/n2cholas/jax-resnet

from flax import nnx
from typing import Callable
from functools import partial
import jax.numpy as jnp
import jax

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
    scale: int = 1
    layers: list = []
    downsample: Callable | None

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride = (1, 1), 
                 downsample: Callable | None = None,
                 *, rngs: nnx.Rngs):
        self.downsample = downsample
        self.layers = [
            nnx.Conv(in_channels, out_channels, (3, 3), strides=stride, use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu,

            nnx.Conv(out_channels, out_channels, (3, 3), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
        ]

class Bottleneck(ResidualBlock):
    scale: int = 4
    layers: list = []
    downsample: Callable | None

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride = (1, 1), 
                 downsample: Callable | None = None,
                 *, rngs: nnx.Rngs):
        self.downsample = downsample
        self.layers = [
            nnx.Conv(in_channels, out_channels, (1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu,

            nnx.Conv(out_channels, out_channels, (3, 3), strides=stride, use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu,

            nnx.Conv(out_channels, out_channels * self.scale, (1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_channels * self.scale, rngs=rngs),
        ]


class ResNet(nnx.Module):
    block: ResidualBlock
    layers: list = []
    output_layer: nnx.Module

    def __init__(self, 
                 block: ResidualBlock, 
                 layers: list[int], 
                 num_output: int, 
                 *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.in_channels = 64
        self.num_output = num_output
        self.block = block
        self.layers = [
            nnx.Sequential(
                nnx.Conv(3, 64, (7, 7), strides=(2, 2), use_bias=False, rngs=rngs),
                nnx.BatchNorm(64, rngs=rngs),
                nnx.relu,
                # partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
            ),
            self.make_layers(64, layers[0]),
            self.make_layers(128, layers[1], stride=(2, 2)),
            self.make_layers(256, layers[2], stride=(2, 2)),
            self.make_layers(512, layers[3], stride=(2, 2)),
        ]
        self.head = nnx.Sequential(
            partial(jnp.mean, axis=(1, 2)),
            nnx.Linear(self.in_channels, self.num_output, rngs=rngs),
            nnx.softmax
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def make_layers(
        self, 
        out_channels: int, 
        num_layers: int, 
        stride = (1, 1)
    ) -> list:
        downsample = None
        if stride != (1, 1) or self.in_channels != out_channels * self.block.scale:
            downsample = nnx.Sequential(
                # Projection Shortcut method
                nnx.Conv(self.in_channels, out_channels * self.block.scale, (1, 1), stride, use_bias=False, rngs=self.rngs),
                nnx.BatchNorm(out_channels * self.block.scale, rngs=self.rngs),
                # TODO: add option to use padding method
            )
        layers = [self.block(self.in_channels, out_channels, stride, downsample, rngs=self.rngs)]
        self.in_channels = out_channels * self.block.scale
        layers += [self.block(self.in_channels, out_channels, rngs=self.rngs)] * (num_layers - 1)
        return nnx.Sequential(*layers)


# Standard ResNets
ResNet18  = partial(ResNet, BasicBlock, [2, 2, 2, 2])
ResNet34  = partial(ResNet, BasicBlock, [3, 4, 6, 3])
ResNet50  = partial(ResNet, Bottleneck, [3, 4, 6, 3])
ResNet101 = partial(ResNet, Bottleneck, [3, 4, 23, 3])
ResNet152 = partial(ResNet, Bottleneck, [3, 8, 36, 3])

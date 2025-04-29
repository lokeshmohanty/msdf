# UNet with ResNet encoder

from flax import nnx
from functools import partial
import jax, jax.numpy as jnp

from msdf.backbones.resnet import ResNet

class UpSampleBlock(nnx.Module):
    layers: list = []

    def __init__(self, in_channels, out_channels, *, rngs: nnx.Rngs):
        self.upconv = nnx.Sequential(
            nnx.ConvTranspose(in_channels, out_channels, (3, 3), strides=(2, 2), rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu,
        )
        self.layers = [
            nnx.Conv(in_channels, out_channels, (3, 3), strides=(2, 2), rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu,

            nnx.Conv(out_channels, out_channels, (3, 3), rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu,
        ]

    def __call__(self, x: jax.Array, x_skip: jax.Array) -> jax.Array:
        x = self.upconv(x)
        x = jnp.concatenate([x, x_skip])
        for layer in self.layers:
            x = layer(x)
        return x

class UNet(ResNet):
    layers: list = []
    decoder_layers: list = []
    head: nnx.Module

    def __init__(self, *args, **kwargs):
        super(UNet, self).__init__(*args, **kwargs)
        self.decoder_layers = []
        self.head = nnx.Sequential(
            nnx.Conv(32, 32, (3, 3), rngs=self.rngs),
            nnx.BatchNorm(32, rngs=self.rngs),
            nnx.relu,

            nnx.Conv(32, 16, (3, 3), rngs=self.rngs),
            nnx.BatchNorm(16, rngs=self.rngs),
            nnx.relu,

            nnx.Conv(16, 16, (1, 1), rngs=self.rngs),
            nnx.softmax,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        stack = []
        for layer in self.layers:
            x = layer(x)
            stack.append(x)
        for layer in self.decoder_layers:
            x = layer(x, stack.pop())
        return self.head(x)

   
# Modified ResNets for CenterNet
ResNetC18  = partial(ResNetC, BasicBlock, [2, 2, 2, 2])
ResNetC34  = partial(ResNetC, BasicBlock, [3, 4, 6, 3])
ResNetC50  = partial(ResNetC, Bottleneck, [3, 4, 6, 3])
ResNetC101 = partial(ResNetC, Bottleneck, [3, 4, 23, 3])
ResNetC152 = partial(ResNetC, Bottleneck, [3, 8, 36, 3])



# Modified ResNet for CenterNet

from flax import nnx
from functools import partial
import jax, jax.numpy as jnp

from msdf.backbones.resnet import ResNet, BasicBlock, Bottleneck


class UpSampleBlock(nnx.Module):
    layers: list = []

    def __init__(self, in_channels, out_channels, *, rngs: nnx.Rngs):
        self.upconv = nnx.Sequential(
            nnx.ConvTranspose(
                in_channels, out_channels, (3, 3), strides=(2, 2), rngs=rngs
            ),
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


class ResNetC(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNetC, self).__init__(*args, **kwargs)
        self.head = nnx.Sequential(
            *self.make_deconv_layers(3, [256] * 3, [(4, 4)] * 3),
            nnx.Conv(256, 64, (3, 3), rngs=self.rngs),
            nnx.relu,
            nnx.Conv(64, self.num_output, (1, 1), rngs=self.rngs),
            nnx.sigmoid,
            lambda x: jnp.transpose(x, (0, 3, 1, 2)),
        )

    def make_deconv_layers(
        self, num_layers: int, out_channels: list, kernels: list
    ) -> list:
        layers: list = []
        for i in range(num_layers):
            layers += [
                nnx.ConvTranspose(
                    self.in_channels,
                    out_channels[i],
                    kernels[i],
                    (2, 2),
                    rngs=self.rngs,
                ),
                nnx.BatchNorm(out_channels[i], rngs=self.rngs),
                nnx.relu,
            ]
            self.in_channels = out_channels[i]
        return layers


# Modified ResNets for CenterNet
ResNetC18 = partial(ResNetC, BasicBlock, [2, 2, 2, 2])
ResNetC34 = partial(ResNetC, BasicBlock, [3, 4, 6, 3])
ResNetC50 = partial(ResNetC, Bottleneck, [3, 4, 6, 3])
ResNetC101 = partial(ResNetC, Bottleneck, [3, 4, 23, 3])
ResNetC152 = partial(ResNetC, Bottleneck, [3, 8, 36, 3])

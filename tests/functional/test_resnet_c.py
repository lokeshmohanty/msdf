import pytest
from flax import nnx
import jax, jax.numpy as jnp

from msdf.backbones.resnet_c import ResNetC18, ResNetC34, ResNetC50, ResNetC101, ResNetC152

@pytest.mark.parametrize("Model, n_params", [
    # TODO: Find true parameters count
    (ResNetC18,  15532428),
    (ResNetC34,  15532428),
    (ResNetC50,  24614796),
    (ResNetC101, 24614796),
    (ResNetC152, 24614796),
])
def test_resnet_c(Model, n_params):
    in_size = (1, 256, 256, 3)
    out_size = (1, 10, 256 // 4, 256 // 4)
    model = Model(out_size[1], rngs=nnx.Rngs(0))
    output = model(jnp.ones(in_size))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(nnx.split(model)[1]))
    assert param_count == n_params
    assert output.shape == out_size


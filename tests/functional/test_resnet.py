import pytest
from flax import nnx
import jax, jax.numpy as jnp

from msdf.backbones.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

@pytest.mark.parametrize("Model, n_params", [
    # TODO: Find true parameters count
    (ResNet18,  11191244),
    (ResNet34,  11191244),
    (ResNet50,  13997516),
    (ResNet101, 13997516),
    (ResNet152, 13997516),
])
def test_resnet(Model, n_params):
    in_size = (1, 256, 256, 3)
    out_size = (1, 10)
    model = Model(out_size[-1], rngs=nnx.Rngs(0))
    output = model(jnp.ones(in_size))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(nnx.split(model)[1]))
    assert param_count == n_params
    assert output.shape == out_size


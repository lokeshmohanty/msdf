from datasets import load_dataset
import numpy as np
import optax
import pytest
from flax import nnx
import jax, jax.numpy as jnp

from msdf.backbones.resnet import ResNet18
from msdf.trainer import Trainer

@pytest.fixture()
def load_data():
    train_indices = jax.random.randint(jax.random.key(0), 200, 0, 100000)
    test_indices = jax.random.randint(jax.random.key(0), 200, 0, 10000)
    ds = { 'train': load_dataset("zh-plus/tiny-imagenet", split="train")[train_indices],
           'test': load_dataset("zh-plus/tiny-imagenet", split="valid")[test_indices], }
    n_classes = 200
    for split in ['train', 'test']:
        ds[split] = [
            np.array([np.array(im.convert('RGB')) for im in ds[split]['image']]),
            np.array(ds[split]['label']),
        ]
        ds[split][0] = jnp.float32(ds[split][0]) / 255
        ds[split][1] = jnp.int32(ds[split][1])
    return ds['train'], ds['test'], n_classes

def test_resnet_c(load_data):
    train, test, n_classes = load_data
    lr, momentum = 5e-3, 0.9
    model = ResNet18(n_classes, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(lr, momentum))
    metrics = nnx.MultiMetric(
        accuracy = nnx.metrics.Accuracy(),
        loss = nnx.metrics.Average('loss'),
    )
    trainer = Trainer(
        model, 
        optax.softmax_cross_entropy_with_integer_labels, 
        optimizer, 
        metrics
    )
    trainer.fit(train, test, is_batched=False)
    accuracy = trainer.metrics_history['train_accuracy']
    assert all(accuracy[i] > accuracy[i+1] for i in range(len(accuracy) - 1))



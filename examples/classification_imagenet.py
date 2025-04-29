import os
import time
import tomllib
from pathlib import Path
from dataclasses import dataclass
from typing import Self

import jax, jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from msdf.backbones.resnet import ResNet18


@dataclass
class Config:
    n_epochs: int = 100
    learning_rate: float = 0.005
    momentum: float = 0.9
    batch_size: int = 1000
    plot_name: Path = "plots/imagenet_training.png"
    checkpoint_dir: Path = "models/imagenet/"

    def __post_init__(self):
        self.plot_name = Path.cwd() / self.plot_name
        self.checkpoint_dir = Path.cwd() / self.checkpoint_dir

    @classmethod
    def load(cls, file: Path) -> Self:
        if not file.exists():
            return cls()

        with open(file, "rb") as f:
            user_config = tomllib.load(f)

        if "env" in user_config.keys():
            for key in user_config["env"].keys():
                os.environ[key] = str(user_config["env"][key])

        if "jax" in user_config.keys():
            for key in user_config["jax"].keys():
                jax.config.update(key, user_config["jax"][key])

        if "imagenet" in user_config.keys():
            return cls(**user_config["imagenet"])
        return cls()


# Load user config
config = Config.load(Path.cwd() / "config.toml")


def load_data():
    print("Loading Imagenet Data")
    ds = {
        "train": load_dataset("zh-plus/tiny-imagenet", split="train"),
        "test": load_dataset("zh-plus/tiny-imagenet", split="valid"),
    }
    n_classes = 200
    for split in ["train", "test"]:
        ds[split] = [
            np.array([np.array(im.convert("RGB")) for im in ds[split]["image"]]),
            np.array(ds[split]["label"]),
        ]
        ds[split][0] = jnp.float32(ds[split][0]) / 255
        ds[split][1] = jnp.int32(ds[split][1])
    print("Imagenet Data Loaded")
    return ds["train"], ds["test"], n_classes


def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch[1]).mean()
    return loss, logits


@nnx.jit
def train_step(model, batch, metrics, optimizer):
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    # metrics.update(loss, logits, labels=batch[1])
    optimizer.update(grads)
    return loss, logits


@nnx.jit
def eval_step(model, batch, metrics):
    loss, logits = loss_fn(model, batch)
    # metrics.update(loss, logits, labels=batch[1])
    return loss, logits


def accuracy(logits, labels):
    n = len(labels)
    correct = jnp.argmax(logits, axis=1) == labels
    return correct.sum() / n


def save_model(model_state, ckpt_dir=config.checkpoint_dir):
    ckpt_mgr = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(ckpt_dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=2,
            keep_checkpoints_without_metrics=False,
            create=True,
        ),
    )
    ckpt_mgr.save(1, args=ocp.args.Composite(state=ocp.args.PyTreeSave(model_state)))
    ckpt_mgr.close()


def load_model(model, ckpt_dir=config.checkpoint_dir) -> nnx.Module:
    if not (Path(ckpt_dir) / "1").exists():
        print("No checkpoint to load")
        return model

    graphdef, model_state = nnx.split(model)
    with ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as read_mgr:
        restored = read_mgr.restore(
            1,
            args=ocp.args.Composite(state=ocp.args.PyTreeRestore(item=model_state)),
        )
        return nnx.merge(graphdef, restored["state"])


def fit_model(model, optimizer, train_data, val_data, n_epochs=config.n_epochs):
    # metrics = nnx.MultiMetric(
    #     accuracy=nnx.metrics.Accuracy(),
    #     loss=nnx.metrics.Average("loss"),
    # )
    metrics = {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
    }
    n = train_data[0].shape[0]
    for epoch in range(n_epochs):
        t_loss, t_acc = 0, 0
        start_time = time.time()

        for batch in zip(*train_data):
            loss, logits = train_step(model, batch, None, optimizer)
            t_acc += accuracy(logits, batch[1])
            t_loss += loss
        print(f"Epoch {epoch + 1} in {time.time() - start_time:.2f} sec: ")
        metrics["train_loss"].append(t_loss / n)
        metrics["train_accuracy"].append(t_acc / n)
        print(f"\tTrain     Loss: {metrics['train_loss'][-1]:.4f}")
        print(f"\tTrain Accuracy: {metrics['train_accuracy'][-1]:.4f}")

        loss, logits = eval_step(model, val_data, None)
        metrics["val_loss"].append(loss)
        metrics["val_accuracy"].append(accuracy(logits, val_data[1]))
        print(f"\t  Val     Loss: {metrics['val_loss'][-1]:.4f}")
        print(f"\t  Val Accuracy: {metrics['val_accuracy'][-1]:.4f}")
    return metrics


def eval_model(model, test_data):
    t_loss, t_acc = 0, 0
    start_time = time.time()
    metrics = {
        "test_accuracy": [],
        "test_loss": [],
    }

    n = test_data[0].shape[0]
    for batch in zip(*test_data):
        loss, logits = eval_step(model, batch, None)
        t_acc += accuracy(logits, batch[1])
        t_loss += loss
    print(
        f"Inference in {(time.time() - start_time)/n:.2f} sec"
        + f" per batch with batch_size: {config.batch_size}: "
    )
    metrics["test_loss"].append(t_loss / n)
    metrics["test_accuracy"].append(t_acc / n)
    print(f"\t    Loss: {metrics['test_loss'][-1]:.4f}")
    print(f"\tAccuracy: {metrics['test_accuracy'][-1]:.4f}")
    return metrics


def plot_metrics(metrics, filename=config.plot_name):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f"Imagenet Classification")

    ax[0][0].set_xlabel("Number of epochs")
    ax[0][0].set_ylabel("train_loss")
    ax[0][0].plot(metrics["train_loss"])
    ax[0][1].set_xlabel("Number of epochs")
    ax[0][1].set_ylabel("train_accuracy")
    ax[0][1].plot(metrics["train_accuracy"])
    ax[1][0].set_xlabel("Number of epochs")
    ax[1][0].set_ylabel("val_loss")
    ax[1][0].plot(metrics["val_loss"])
    ax[1][1].set_xlabel("Number of epochs")
    ax[1][1].set_ylabel("val_accuracy")
    ax[1][1].plot(metrics["val_accuracy"])

    fig.tight_layout()
    fig.savefig(filename)


# Load tiny-imagenet data
train_data, test_data, n_classes = load_data()

# Preprocess data
train_data[0] = train_data[0].reshape(-1, config.batch_size, 64, 64, 3)
train_data[1] = train_data[1].reshape(-1, config.batch_size)
test_data[0] = test_data[0].reshape(-1, config.batch_size, 64, 64, 3)
test_data[1] = test_data[1].reshape(-1, config.batch_size)
val_data = test_data[0][0], test_data[1][0]

# Define the model
model = ResNet18(n_classes, rngs=nnx.Rngs(0))
model = load_model(model)

# Train the model
optimizer = nnx.Optimizer(model, optax.adamw(config.learning_rate, config.momentum))
train_metrics = fit_model(model, optimizer, train_data, val_data)
save_model(nnx.state(model))

# Visualize model training
plot_metrics(train_metrics)

# Evaluate the model
eval_metrics = eval_model(model, test_data)

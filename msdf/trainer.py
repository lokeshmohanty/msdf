from dataclasses import dataclass
from flax import nnx
import orbax.checkpoint as ocp
from typing import Callable
import jax
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)
logging.getLogger("absl").setLevel(logging.WARNING)


@dataclass
class Trainer:
    model: nnx.Module
    loss_fn: Callable[[jax.Array, jax.Array], tuple[float, jax.Array]]
    optimizer: nnx.Optimizer
    metrics: nnx.MultiMetric
    metrics_history = {}
    ckpt_dir = "/tmp/msdf-checkpoints/"

    def __post_init__(self):
        logger.info("Initialize Trainer")
        self.loss_fn = gen_loss_fn(self.loss_fn)
        self.train_step = nnx.jit(gen_train_step(self.loss_fn))
        self.eval_step = nnx.jit(gen_eval_step(self.loss_fn))
        for metric in self.metrics._metric_names:
            self.metrics_history[f"train_{metric}"] = []
            self.metrics_history[f"test_{metric}"] = []
        self.ckpt_dir = ocp.test_utils.erase_and_create_empty(self.ckpt_dir)
        self.checkpointer = ocp.StandardCheckpointer()

    def fit(self, train, test, n_epochs=100, is_batched=True):
        if not is_batched:
            train, test = map(batch_data, [train, test])
        logger.info("Training Started")
        for epoch in range(1, n_epochs + 1):
            for batch in zip(*train):
                self.train_step(self.model, batch, self.metrics, self.optimizer)

            logger.info(f"Epoch {epoch:3d}:")
            for metric, value in self.metrics.compute().items():
                self.metrics_history[f"train_{metric}"].append(value)
                logger.info(f"train_{metric}: {value}")
            self.metrics.reset()

            for batch in zip(*test):
                self.eval_step(self.model, batch, self.metrics)

            for metric, value in self.metrics.compute().items():
                self.metrics_history[f"test_{metric}"].append(value)
                logger.info(f"test_{metric}: {value}")
            self.metrics.reset()
        logger.info("Training Completed")

    def save(self):
        _, state = nnx.split(self.model)
        pure_dict_state = nnx.to_pure_dict(state)
        self.checkpointer.save(self.ckpt_dir / "pure_dict", pure_dict_state)

    def load(self):
        restored_pure_dict = self.checkpointer.restore(self.ckpt_dir / "pure_dict")
        abstract_model = nnx.eval_shape(lambda: self.model)
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
        self.model = nnx.merge(graphdef, abstract_state)


def gen_loss_fn(loss_fn):
    def new_loss_fn(model, batch):
        logits = model(batch[0])
        loss = loss_fn(logits, batch[1]).mean()
        return loss, logits

    return new_loss_fn


def gen_train_step(loss_fn):
    def train_step(model, batch, metrics, optimizer):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        metrics.update(loss=loss, logits=logits, labels=batch[1])
        optimizer.update(grads)

    return train_step


def gen_eval_step(loss_fn):
    def eval_step(model, batch, metrics):
        loss, logits = loss_fn(model, batch)
        metrics.update(loss=loss, logits=logits, labels=batch[1])

    return eval_step


def batch_data(data, batch_size=32):
    n = data[0].shape[0] // batch_size
    n_features = data[0].shape[1:]
    if n == 0:
        return (
            data[0].reshape(1, *data[0].shape),
            data[1].reshape(1, data[1].shape[0]),
        )
    return (
        data[0][: n * batch_size].reshape(-1, batch_size, *n_features),
        data[1][: n * batch_size].reshape(-1, batch_size),
    )

from flax import nnx
import optax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import cars
from centernet import CenterNet
from backbones.resnet import make_resnet
from utils import setup_logger, visualize_detections

logger = setup_logger(__name__)

IMAGE_SIZE = (512, 512)
OUTPUT_STRIDE = 4
NUM_CLASSES = 2
BATCH_SIZE = 16

def get_data(centernet):
    logger.info("Generate data")
    images, bboxes = cars.load_data(limit=BATCH_SIZE, image_size=IMAGE_SIZE)

    logger.info("Process data")
    images = jnp.asarray(images) / 255.0
    targets = jnp.asarray([centernet.bboxes_to_heatmaps(bbox) for bbox in bboxes])

    images = images.reshape(-1, BATCH_SIZE, *IMAGE_SIZE, 3)
    targets = targets.reshape(-1, BATCH_SIZE, NUM_CLASSES + 4, 
                              IMAGE_SIZE[0] // OUTPUT_STRIDE, 
                              IMAGE_SIZE[1] // OUTPUT_STRIDE
                              )
    return images, targets

def train(images, targets):
    learning_rate = 0.005
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        # iou=nnx.metrics.Average("iou"),
        loss=nnx.metrics.Average("loss"),
    )


    @nnx.jit
    def loss_fn(model: nnx.Module, batch):
        outputs = model(batch[0])
        loss = nnx.vmap(centernet.loss_fn)(outputs, batch[1]).mean()
        return loss, outputs


    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grads = grad_fn(model, batch)
        metrics.update(loss=loss)  # In-place updates.
        optimizer.update(grads)  # In-place updates.


    @nnx.jit
    def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
        loss, _ = loss_fn(model, batch)
        metrics.update(loss=loss)  # In-place updates.

    metrics_history = {
        "train_loss": [],
        # 'train_iou': [],
        # 'test_loss': [],
        # 'test_iou': [],
    }

    logger.info("Training started")
    for epoch in tqdm(range(100)):
        for batch in zip(images, targets):
            train_step(model, optimizer, metrics, batch)

        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)
        metrics.reset()

        # for test_batch in test_ds.as_numpy_iterator():
        #   eval_step(model, metrics, test_batch)

        # for metric, value in metrics.compute().items():
        #   metrics_history[f'test_{metric}'].append(value)
        # metrics.reset()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title("Loss")
        # ax2.set_title("Intersection over Union (IOU)")
        ax1.plot(metrics_history[f"train_loss"], label=f"train_loss")
        # ax2.plot(metrics_history[f"train_iou"], label=f"train_iou")
        ax1.legend()
        # ax2.legend()
        logger.info(f"Traininig loss: {metrics_history['train_loss'][-1]}")
        plt.savefig("plots/train-cars.png")
        # plt.show()
    logger.info("Training done")


if __name__ == "__main__":
    centernet = CenterNet(image_size=IMAGE_SIZE, output_stride=OUTPUT_STRIDE)
    images, targets = get_data(centernet)

    logger.info("Initialize model")
    model = make_resnet(101)(NUM_CLASSES + 4, rngs=nnx.Rngs(0))
    train(images, targets)

    logger.info("Evaluate the trained model")
    model.eval()  # Switch to evaluation mode.

    logger.info("Plot visualization")

    outputs = model(images[0][:2])
    bbox1 = centernet.heatmaps_to_bboxes(outputs[0])
    bbox2 = centernet.heatmaps_to_bboxes(outputs[1])
    visualize_detections(images[0][1], bbox1, cars.LABELS, filename="plots/detections1.png")
    visualize_detections(images[0][1], bbox2, cars.LABELS, filename="plots/detections2.png")

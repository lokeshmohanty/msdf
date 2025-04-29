from flax import nnx
import optax
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
import logging

from msdf.data import cars
from msdf.centernet import CenterNet
from msdf.backbones.resnet_c import ResNetC18
from msdf.trainer import Trainer
from msdf.utils import visualize_detections

IMAGE_SIZE = 512
OUTPUT_STRIDE = 4
NUM_CLASSES = 2
BATCH_SIZE = 16

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def get_data(centernet):
    logger.info("Generate data")
    images, bboxes = cars.load_data(limit=15 * BATCH_SIZE, image_size=IMAGE_SIZE)

    logger.info("Process data")
    images = jnp.asarray(images) / 255.0
    targets = jnp.asarray([centernet.bboxes_to_heatmaps(bbox) for bbox in bboxes])

    images = images.reshape(-1, BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    targets = targets.reshape(
        -1,
        BATCH_SIZE,
        NUM_CLASSES + 4,
        IMAGE_SIZE // OUTPUT_STRIDE,
        IMAGE_SIZE // OUTPUT_STRIDE,
    )
    return (
        (images[: 13 * BATCH_SIZE], targets[: 13 * BATCH_SIZE]),
        (images[13 * BATCH_SIZE :], targets[13 * BATCH_SIZE :]),
    )

    # plt.figure(1)
    # _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # ax1.set_title("Loss")
    # # ax2.set_title("Intersection over Union (IOU)")
    # ax1.plot(metrics_history[f"train_loss"], label=f"train_loss")
    # # ax2.plot(metrics_history[f"train_iou"], label=f"train_iou")
    # ax1.legend()
    # # ax2.legend()
    # logger.info(f"Traininig loss: {metrics_history['train_loss'][-1]}")
    # plt.savefig("plots/train-cars.png")
    # # plt.show()


if __name__ == "__main__":

    centernet = CenterNet(
        image_size=IMAGE_SIZE, output_stride=OUTPUT_STRIDE, n_classes=2
    )
    train_data, test_data = get_data(centernet)
    model = ResNetC18(NUM_CLASSES + 4, rngs=nnx.Rngs(0))

    learning_rate = 0.005
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        # iou=nnx.metrics.Average("iou"),
        loss=nnx.metrics.Average("loss"),
    )
    trainer = Trainer(
        model,
        nnx.vmap(centernet.loss_fn),
        optimizer,
        metrics,
    )
    trainer.fit(train_data, test_data)

    logger.info("Evaluate the trained model")
    trainer.model.eval()  # Switch to evaluation mode.

    logger.info("Plot visualization")

    outputs = model(train_data[0][0][:2])
    bbox1 = centernet.heatmaps_to_bboxes(outputs[0])
    bbox2 = centernet.heatmaps_to_bboxes(outputs[1])
    visualize_detections(
        train_data[0][0][1], bbox1, cars.LABELS, filename="plots/detections1.png"
    )
    visualize_detections(
        train_data[0][0][1], bbox2, cars.LABELS, filename="plots/detections2.png"
    )

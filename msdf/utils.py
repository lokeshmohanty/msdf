import matplotlib.pyplot as plt
import logging

def setup_logger(name: str, handlers: list[str] = ["console", "file"], level = logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if "console" in handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    elif "file" in handlers:
        fh = logging.FileHandler(f'logs/{name}.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Jaccard index: IoU (Metric)
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    intersection = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    return intersection / union


def visualize_detections(image, boxes, labels, filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image)
    ax2.imshow(image)

    for box in boxes:
        l, x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            xy=(x1, y1),
            width=x2 - x1,
            height=y2 - y1,
            linewidth=2,
            edgecolor="r",
            fill=False,
        )
        ax2.add_patch(rect)
        ax2.text(
            x1,
            y1,
            labels[l] if isinstance(l, int) else labels[l[0]],
            fontsize=12,
            color="w",
            bbox=dict(facecolor="r", lw=0),
        )

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

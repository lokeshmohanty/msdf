from PIL import Image
import optax
import jax
import numpy as np
import jax.numpy as jnp


class CenterNet:
    def __init__(
        self, image_size=(256, 256), output_stride=4, key=jax.random.PRNGKey(42)
    ):
        self.image_size = image_size
        self.r = output_stride

    def bbox_to_center(self, box):
        """
        Convert bounding box to center and size format

        Returns:
            tuple: (center, size)
        """
        (x1, y1), (x2, y2) = box
        return ((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1)

    def center_to_bbox(self, center, size):
        """
        Convert center and size to bounding box format

        Returns:
            tuple: ((x1, y1), (x2, y2))
        """
        x, y = center
        w, h = size
        return (x - w / 2, y - h / 2), (x + w / 2, y + h / 2)

    def bboxes_to_heatmaps(self, bboxes):
        """
        Convert bounding boxes to heatmaps

        Returns:
            np.array: array of heatmaps
        """
        n_classes = 2
        heatmaps = np.zeros(
            (
                n_classes + 4,
                self.image_size[0] // self.r,
                self.image_size[1] // self.r,
            )
        )
        for bbox in bboxes:
            c, x1, y1, x2, y2 = (
                bbox[0],
                bbox[1] / self.r,
                bbox[2] / self.r,
                bbox[3] / self.r,
                bbox[4] / self.r,
            )
            sigma = (x2 - x1 + y2 - y1) / 6
            center = (x1 + x2) / 2, (y1 + y2) / 2
            heatmaps[-4][int(center[0]), int(center[1])] = (
                bbox[1] + bbox[3]
            ) / 2 - x1  # offset x
            heatmaps[-3][int(center[0]), int(center[1])] = (
                bbox[2] + bbox[4]
            ) / 2 - y1  # offset y
            heatmaps[-2][int(center[0]), int(center[1])] = x2 - x1  # width
            heatmaps[-1][int(center[0]), int(center[1])] = y2 - y1  # height
            for i in range(int(y1), int(y2)):
                for j in range(int(x1), int(x2)):
                    heatmaps[c][i, j] = jnp.exp(
                        -((round(center[0]) - j) ** 2 + (round(center[1]) - i) ** 2)
                        / (2 * sigma**2)
                    )
        return jnp.asarray(heatmaps)

    def heatmaps_to_bboxes(self, heatmaps):
        """
        Convert heatmaps to bounding boxes

        Returns:
            list: list of bounding boxes with detection confidence
        """
        size_map = heatmaps[-2:]
        offset_map = heatmaps[-4:-2]
        keypoints = []
        for c, heatmap in enumerate(heatmaps[:-4]):
            peaks = []
            shape = heatmap.shape
            is_local_maxima = (
                lambda x, y: heatmap[x, y]
                == heatmap[x - 1 : x + 2, y - 1 : y + 2].max()
            )
            for i in range(1, shape[0] - 1):
                for j in range(1, shape[1] - 1):
                    if is_local_maxima(i, j):
                        peaks.append((i, j, heatmap[i, j]))
            peaks = sorted(peaks, key=lambda x: x[2])[:100]

            def keypoint(i, j, c, hm, offset_map, size_map):
                return (
                    (c, hm[i, j]),
                    (i, j),
                    (offset_map[0][i, j].item(), offset_map[1][i, j].item()),
                    (size_map[0][i, j].item(), size_map[1][i, j].item()),
                )

            keypoints = [
                keypoint(i, j, c, heatmap, offset_map, size_map) for i, j, _ in peaks
            ]
        return [
            (
                (c, v),
                x - w / 2 + offset[0],
                y - h / 2 + offset[1],
                x + w / 2 + offset[0],
                y + h / 2 + offset[1],
            )
            for ((c, v), (x, y), offset, (w, h)) in keypoints
        ]

    def _zero_non_local_maxima(self, x):
        s = x.shape
        x1 = jnp.roll(x, shift=1, axis=0).at[0].set(0)
        x2 = jnp.roll(x, shift=-1, axis=0).at[s[0] - 1].set(0)
        x3 = jnp.roll(x, shift=1, axis=1).at[:, 0].set(0)
        x4 = jnp.roll(x, shift=-1, axis=1).at[:, s[1] - 1].set(0)
        x5 = jnp.roll(x1, shift=1, axis=1).at[:, 0].set(0)
        x6 = jnp.roll(x1, shift=-1, axis=1).at[:, s[1] - 1].set(0)
        x7 = jnp.roll(x2, shift=1, axis=1).at[:, 0].set(0)
        x8 = jnp.roll(x2, shift=-1, axis=1).at[:, s[1] - 1].set(0)

        return jnp.where(
            (x > x1)
            & (x > x2)
            & (x > x3)
            & (x > x4)
            & (x > x5)
            & (x > x6)
            & (x > x7)
            & (x > x8),
            x,
            0,
        )

    def loss_fn(self, output, target):
        alpha, beta = 2, 4
        lambda_offset = 1
        lambda_size = 0.1
        pred_y, y = output[:-4], target[:-4]
        pred_y_max = self._zero_non_local_maxima(pred_y).max(axis=0)

        pixel_wise_loss = jnp.abs(
            jnp.where(
                y == 1,
                (1 - pred_y) ** alpha * jnp.log(pred_y),
                (1 - y) ** beta * pred_y**alpha * jnp.log(1 - pred_y),
            )
        ).mean()
        offset_loss = jnp.where(
            pred_y_max == 0,
            0,
            jnp.abs(output[-4] - target[-4]) + jnp.abs(output[-3] - target[-3]),
        ).sum()
        size_loss = jnp.where(
            pred_y_max == 0,
            0,
            jnp.abs(output[-2] - target[-2]) + jnp.abs(output[-1] - target[-1]),
        ).sum()

        return pixel_wise_loss + lambda_offset * offset_loss + lambda_size * size_loss

    def iou(self, outputs, targets):
        intersection = 1e-8
        union = 1e-8
        for i in range(outputs.shape[0]):
            output = self.heatmaps_to_bboxes(outputs[i])
            target = self.heatmaps_to_bboxes(targets[i])

            for o, t in zip(output, target):
                area_o = (o[3] - o[1]) * (o[4] - o[2])
                area_t = (t[3] - t[1]) * (t[4] - t[2])
                area_inter = max(0, min(o[3], t[3]) - max(o[1], t[1])) * max(
                    0, min(o[4], t[4]) - max(o[2], t[2])
                )
                area_union = area_o + area_t - area_inter
                intersection += area_inter
                union += area_union
        return intersection / union


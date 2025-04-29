import numpy as np
import jax, jax.numpy as jnp


class CenterNet:
    def __init__(self, image_size: int, output_stride: int, n_classes: int):
        """
        Sets:
            input_size and output_size of the network
            number of output maps is
                number of classes (keypoint types)
                + 4(2 for size and 2 for offset, regression maps)

        Args:
            image_size: input size of image
            output_stride: output stride of the network
            n_classes: number of classes (keypoint types)
        """
        self.n_classes = n_classes
        self.r = output_stride
        self.in_size = image_size

        self.n_maps = n_classes + 4  # 2 for size and 2 for offset
        self.out_size = self.in_size // self.r

    def bbox_to_center(self, box):
        """
        Convert bounding box to center and size format

        Args:
            box: (x_min, y_min, x_max, y_max)

        Returns:
            tuple: (center, size)
        """
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1)

    def center_to_bbox(self, center, size):
        """
        Convert center and size to bounding box format

        Args:
            center: (x, y)
            size: (width, height)

        Returns:
            tuple: (x_min, y_min, x_max, y_max)
        """
        x, y = center
        w, h = size
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2

    def bboxes_to_heatmaps(self, bboxes):
        """
        Convert bounding boxes to heatmaps
        output maps:
            one for each class,
            width,
            height,
            offset_x,
            offset_y

        Returns:
            np.array: array of heatmaps
        """
        heatmaps = np.zeros((self.n_maps, self.out_size, self.out_size))
        for bbox in bboxes:
            c, x1, y1, x2, y2 = (
                int(bbox[0]),
                bbox[1] / self.r,
                bbox[2] / self.r,
                bbox[3] / self.r,
                bbox[4] / self.r,
            )
            _center, (width, height) = self.bbox_to_center((x1, y1, x2, y2))
            center = int(_center[0]), int(_center[1])
            offset = _center[0] - center[0], _center[1] - center[1]
            sigma = (width + height) / 6
            heatmaps[-4][center] = width
            heatmaps[-3][center] = height
            heatmaps[-2][center] = offset[0]
            heatmaps[-1][center] = offset[1]
            for i in range(int(x1), int(x2)):
                for j in range(int(y1), int(y2)):
                    val = center[0] - i, center[1] - j
                    heatmaps[c][i, j] = np.exp(
                        -np.power(val, 2).sum() / (2 * sigma**2 + 1e-5)
                    )
        return heatmaps

    def heatmaps_to_bboxes(self, heatmaps):
        """
        Convert heatmaps to bounding boxes

        Returns:
            list: list of bounding boxes with detection confidence
        """
        offset_map = heatmaps[-2:]  # offset x, offset y
        size_map = heatmaps[-4:-2]  # width, height
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
                x + offset[0] - w / 2,
                y + offset[1] - h / 2,
                x + offset[0] + w / 2,
                y + offset[1] + h / 2,
            )
            for ((c, v), (x, y), offset, (w, h)) in keypoints
        ]

    def _mask_non_local_maxima(self, x):
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
        pred_y_max = self._mask_non_local_maxima(pred_y).max(axis=0)

        pixel_wise_loss = jnp.abs(
            jnp.where(
                y == 1,
                (1 - pred_y) ** alpha * jnp.log(pred_y + 1e-6),
                (1 - y) ** beta * pred_y**alpha * jnp.log(1 - pred_y + 1e-6),
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

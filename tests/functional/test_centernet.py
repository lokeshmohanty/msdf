import pytest
import numpy as np
import jax.numpy as jnp
from msdf.centernet import CenterNet

@pytest.fixture
def centernet():
    return CenterNet(image_size=512, output_stride=4, n_classes=2)

def test_init(centernet):
    assert centernet.n_classes == 2
    assert centernet.r == 4
    assert centernet.in_size == 512
    assert centernet.n_maps == 6  # 2 classes + 4 (size and offset)
    assert centernet.out_size == 128  # 512 / 4

def test_bbox_to_center(centernet):
    box = (10, 20, 30, 40)
    center, size = centernet.bbox_to_center(box)
    assert center == (20, 30)
    assert size == (20, 20)

def test_center_to_bbox(centernet):
    center = (20, 30)
    size = (20, 20)
    box = centernet.center_to_bbox(center, size)
    assert box == (10, 20, 30, 40)

def test_bboxes_to_heatmaps(centernet):
    bboxes = [(0, 0, 0, 64, 64)]  # class, x1, y1, x2, y2
    heatmaps = centernet.bboxes_to_heatmaps(bboxes)
    
    assert heatmaps.shape == (6, 128, 128)  # n_maps, out_size, out_size
    assert heatmaps[0].max() <= 1.0  # class heatmap
    assert heatmaps[-4].max() == 16.0  # width (64/4)
    assert heatmaps[-3].max() == 16.0  # height (64/4)
    # Check if offset is calculated
    # assert np.any(heatmaps[-2] != 0) or np.any(heatmaps[-1] != 0)

def test_heatmaps_to_bboxes(centernet):
    # Create a simple heatmap with one detection
    heatmaps = np.zeros((6, 128, 128))
    heatmaps[0, 64, 64] = 0.9  # class confidence
    heatmaps[-4, 64, 64] = 16  # width
    heatmaps[-3, 64, 64] = 16  # height
    heatmaps[-2, 64, 64] = 0.5  # offset_x
    heatmaps[-1, 64, 64] = 0.5  # offset_y
    
    # bboxes = centernet.heatmaps_to_bboxes(heatmaps)
    # assert len(bboxes) > 0
    # ((label, conf), x1, y1, x2, y2) = bboxes[0]
    # assert label == 0
    # assert conf == 0.9
    # # Check if coordinates are reasonable (accounting for stride and offset)
    # assert 50 < x1 < 70  # rough range considering stride=4
    # assert 50 < y1 < 70
    # assert 60 < x2 < 80
    # assert 60 < y2 < 80

def test_mask_non_local_maxima(centernet):
    x = jnp.array([[0.1, 0.2, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.1, 0.3, 0.2]])
    result = centernet._mask_non_local_maxima(x)
    expected = jnp.array([[0, 0, 0],
                         [0, 0.5, 0],
                         [0, 0, 0]])
    assert jnp.array_equal(result, expected)

def test_loss_fn(centernet):
    output = jnp.ones((6, 128, 128)) * 0.5
    target = jnp.zeros((6, 128, 128))
    target = target.at[0, 64, 64].set(1.0)  # Set one positive example
    
    loss = centernet.loss_fn(output, target)
    assert loss > 0
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()  # scalar

def test_iou(centernet):
    outputs = np.zeros((1, 6, 128, 128))
    targets = np.zeros((1, 6, 128, 128))
    
    # Add similar detections
    outputs[0, 0, 64, 64] = 0.9
    outputs[0, -4, 64, 64] = 16
    outputs[0, -3, 64, 64] = 16
    
    targets[0, 0, 64, 64] = 0.9
    targets[0, -4, 64, 64] = 16
    targets[0, -3, 64, 64] = 16
    
    iou = centernet.iou(outputs, targets)
    assert 0 < iou <= 1.0
    assert isinstance(iou, float)

if __name__ == "__main__":
    pytest.main()

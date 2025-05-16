# VoxelNet Implementation
#
# Paper Title
# VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
 
from flax import nnx
import optax
import jax.numpy as jnp
import jax
from functools import partial
# from tqdm import tqdm
# import matplotlib.pyplot as plt


# Data: [x, y, z, r]^N
# N datapoints with their 3D position and reflectance
# Pointcloud dimensions: D(depth) x H(height) x W(width)
data = []

# Convert Data into voxels
# Create a K x T x 7 tensor for storing voxel input features
# K: max number of non empty voxels 
# T: max number of points per voxel (points chosen randomly)
# 7: [x, y ,z, r, x - vx, y - vy, z - vz], where (vx, vy, vz) 
#   is the centroid of all points in that voxel


# Stacked VFE (Voxel Feature Encoding) layers
# FCN: [Linear, BN, ReLU]
# Concatenate Element(voxel) wise Maxpool
class VFE(nnx.Module):
    def __init__(self, cin, cout, *, rngs=rngs):
        self.cin = cin
        self.cout = cout
        self.fcn = nnx.Sequential([
            nnx.Linear(cin, cout, rngs=rngs),
            nnx.BatchNorm(cout, rngs=rngs),
            nnx.relu,
        ])
        self.concatMax = lambda x: jnp.c_[x, jnp.c_[x.shape[0] * x.max(axis=0)]]

    # Args:
    #   x: voxel ([point]^T)
    def __call__(self, x):
        x = self.fcn(x)
        x = self.concatMax(x)
        return x

# Convolutional Middle Layers
# [ConvMD, BN, ReLU]

# Region Proposal Network
class RPN(nnx.Module):
    def __init__(self, rngs):
        self.rngs = rngs
        self.block1 = [
            nnx.Conv(128, 128, (3, 3), (2, 2), rngs=rngs),
            *([nnx.Conv(128, 128, (3, 3), (1, 1), rngs=rngs)] * 3),
        ]
        self.block2 = [
            nnx.Conv(128, 128, (3, 3), (2, 2), rngs=rngs),
            *([nnx.Conv(128, 128, (3, 3), (1, 1), rngs=rngs)] * 5),
        ]
        self.block3 = [
            nnx.Conv(128, 256, (3, 3), (2, 2), rngs=rngs),
            *([nnx.Conv(256, 256, (3, 3), (1, 1), rngs=rngs)] * 5),
        ]

    def __call__(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x = jnp.c_[
            nnx.ConvTranspose(128, 256, (3, 3), (1, 1), rngs=self.rngs)(x1),
            nnx.ConvTranspose(128, 256, (2, 2), (2, 2), rngs=self.rngs)(x2),
            nnx.ConvTranspose(256, 256, (4, 4), (4, 4), rngs=self.rngs)(x3),
        ]
        prob_score_map = nnx.Conv(768,  2, (1, 1), (1, 1), rngs=self.rngs)(x)
        regression_map = nnx.Conv(768, 14, (1, 1), (1, 1), rngs=self.rngs)(x)

        return prob_score_map, regression_map


# Loss Function


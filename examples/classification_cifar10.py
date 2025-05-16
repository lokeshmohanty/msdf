import polars as pl
import numpy as np
import io
from PIL import Image
import time

splits = {
    "train": "plain_text/train-00000-of-00001.parquet",
    "test": "plain_text/test-00000-of-00001.parquet",
}
df_train = pl.read_parquet("hf://datasets/uoft-cs/cifar10/" + splits["train"])
df_test = pl.read_parquet("hf://datasets/uoft-cs/cifar10/" + splits["test"])


def binary_to_numpy(binary_data):
    return np.array(Image.open(io.BytesIO(binary_data)))


def get_data(df):
    start_time = time.time()
    binary_series = df.select(pl.col("img").struct.field("bytes")).to_series()
    images_list = binary_series.map_elements(binary_to_numpy, return_dtype=pl.Object)
    images_array = np.stack(images_list.to_list())
    labels_array = df.select(pl.col("label")).to_numpy().flatten()
    print(f"Time (pl): {time.time() - start_time}")
    return (images_array, labels_array)


def get_data_np(df):
    start_time = time.time()
    binary_images = df.select(pl.col("img").struct.field("bytes")).to_numpy().flatten()
    images_array = np.array([binary_to_numpy(img) for img in binary_images])
    labels_array = df.select(pl.col("label")).to_numpy().flatten()
    print(f"Time (np): {time.time() - start_time}")
    return (images_array, labels_array)


a = get_data(df_test)
b = get_data_np(df_test)
c = get_data(df_train)
d = get_data_np(df_train)

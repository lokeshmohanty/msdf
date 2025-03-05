# Object Detection by Multi-Sensor Data Fusion

## Cars Dataset
  
  Reference: [kaggle](https://www.kaggle.com/code/killa92/cars-object-tracking-with-fasterrcnn)

## NuScenes Dataset

  Reference: [NuScenes](https://www.nuscenes.org/data/)


- Download the minimized version and extract

    ```shell
    mkdir -p /data/sets/nuscenes
    wget https://www.nuscenes.org/data/v1.0-mini.tgz
    tar -xf v1.0-mini.tgz -C /data/sets/nuscenes
    ````
- Run [`CenterNet.ipynb`](./notebooks/CenterNet.ipynb) notebook

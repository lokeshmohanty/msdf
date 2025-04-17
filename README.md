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

## Troubleshooting

- `FigCanvas..` error occurs due to `nuscenes-devkit` not being compatible
  with the latest versions of `matplotlib`
- In the virtual environment, go to the file 
  `.venv/lib/python3.10/site-packages/nuscenes/nuscenes.py` and in lines
  909 and 911, change `fig.canvas.set_window_title` to
  `fig.canvas.manager.set_window_title`

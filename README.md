# Stereo Visual Odometry (OAK-D)

This repository is a C++ OpenCV implementation of Stereo Visual Odometry based on [ZhenghaoFei/visual_odom](https://github.com/ZhenghaoFei/visual_odom) with support to OAK-D cameras.

### Requirements

- [OpenCV](https://opencv.org/)
- [depthai-core](https://github.com/luxonis/depthai-core/tree/main)

### Dataset

This project also works with the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset

## Compile and Run

Clone the repository, then

**Compiling**

```bash
cmake -S. -Sbuild
cmake --build build
```

**Running**

```bash
# when running using an OAK-D stereo camera
./build/oakdvio [path-to-calibration-file].yaml
# when running using the KITTI dataset
./build/oakdvio /kitti-dataset/sequences/00 ./src/violib/calibration/kitti00.yaml
```



## Using Docker

**Building**

```bash
docker compose build
```

**Running**

```bash
docker compose up -d
docker compose exec vio ./build/oakdvio /kitti-dataset/sequences/00 ./src/violib/calibration/kitti00.yaml 
```

> To be able to run using the KITTI dataset you must first check the configured volumes at `docker-compose.yaml`

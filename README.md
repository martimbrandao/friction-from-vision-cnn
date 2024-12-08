# Friction from vision CNN

This repository holds a ros node and a standalone demo to estimate *friction from vision* using material recognition CNNs and known per-material probability distributions of friction.
The work was finalist for Best Paper Award at the IEEE conference Humanoids 2016, where it was used to help plan slippage-safe motion for humanoid robots.

## Paper

If you use this in your research, please cite

> Martim Brandao, Yukitoshi Minami Shiguematsu, Kenji Hashimoto, and Atsuo Takanishi, "**Material recognition CNNs and hierarchical planning for biped robot locomotion on slippery terrain**", in *16th IEEE-RAS International Conference on Humanoid Robots*, 2016, pp. 81-88.

The paper is available [here](http://www.martimbrandao.com/papers/Brandao2016-humanoids-planning.pdf).

## Instructions

### Caffe

Please download and compile [caffe-segnet](https://github.com/alexgkendall/caffe-segnet).
If you have want to use (recent) cuDNN acceleration, use [this branch](https://github.com/TimoSaemann/caffe-segnet-cudnn5) instead.

Then, edit *scripts/test_webcam_demo.py* and *scripts/rosnode.py* so that this line
```
caffe_root = '/home/martim/workspace/cv/SegNet/caffe-segnet-cudnn5/'
```
points to the correct path for caffe-segnet.

### Data

Download the network weights from [here](https://drive.google.com/file/d/1DC4B5MNsGpq-z5GBkVagBIxl38Kvfh98/view?usp=drive_link) and place them under the folder "data".

### Usage (standalone)

This demo is adapted from [SegNet](https://github.com/alexgkendall/SegNet-Tutorial)'s demo and assumes you have a webcam recognized by OpenCV.
```
cd scripts
bash test_webcam_demo.py
```

### Usage (rosnode)

```
rosrun friction_from_vision rosnode.py
```

You will probably want to edit the ros topic names, which are written on the script directly.

NOTE: for some reason, on my computer caffe-segnet-cudnn5 runs very slow on the rosnode but not on demo (10 seconds vs 100ms). If you run into the same problem, I recommend that you use the original caffe-segnet, compiled without cuDNN support.

### Parameters

The way friction quantiles are computed is by searching the CDF of friction on discretized intervals (called *Q_search_tol* on the code) until the desired quantile probablity is found (*Q_quantile* on the code). This is done on the network directly by adding some extra layers after the material recognition SegNet.

To change the default values for these, please edit *data/segnet_webdemo2_friction.prototxt*:
```
layer {
  name: "pdist"
  bottom: "cdf"
  top: "pdist"
  type: "Power"
  power_param {
    power: 2
    scale: 1
    shift: -0.05 # this is -p (so that we search for p-quantiles)
  }
}
(...)
layer {
  name: "Q"
  bottom: "Qind"
  top: "Q"
  type: "Power" # using power for compatibility with old caffe versions which dont have scale layers
  power_param {
    power: 1
    scale: 0.025 # this is the discretization used during quantile search
    shift: 0
  }
}
```

# JARVIS-HybridNet

<p align="center">
<img src="docs/banner_hybridnet.png" alt="banner" width="70%"/>
</p>

JARIS-HybridNet is a Python library for precise multi-view markerless 3D motion capture in complex environments. Our hybrid 2D-and 3D-CNN pose estimation network is designed to provide precise and robust tracking even under heavy occlusions.

The primary goal of JARVIS to make markerless 3D pose estimation easy to use and quick to implement. With that in mind our network architecture was specifically designed to work with small manually annotated datasets. To make the process of acquiring multi-camera recordings and annotated training data as painless as possible we also provide our [AcquisitionTool]() and our [AnnotationTool](). Assuming you have a set of [FLIR Machine Vision Cameras](https://www.flir.eu/iis/machine-vision/) those tools will enable you to set up your motion capture pipeline without writing a single line of code.  


Check out our [Getting Started Guide](https://jarvis-mocap.github.io/jarvis-docs//2021-10-28-gettingstarted.html) if you want to learn more.  

<p align="center">
<img src="docs/Pytorch_Vid.gif" alt="banner" width="70%"/>
</p>

## Install Instructions

- Clone the repository with
```
git clone https://github.com/JARVIS-MoCap/JARVIS-HybridNet.git
cd JARVIS-HybridNet
```

- Make sure [Anaconda](https://www.anaconda.com/products/individual) is installed on your machine.

- Setup the jarvis Anaconda environment and activate it
```
conda create -n jarvis python=3.9  pytorch=1.10.1 torchvision cudatoolkit=11.3 notebook  -c pytorch
conda activate jarvis
```

- Make sure your setuptools package is up to date \
  `pip install -U setuptools==59.5.0`

- Install JARVIS
  `pip install -e .`
  
 - To be able to use the optional TensorRT acceleration install [Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT) and the [TensorRT] pip package with:
```
pip install nvidia-pyindex
pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
pip install --upgrade nvidia-tensorrt
```
- You will also have to add `libnvinfer.so` to the `PATH` variable

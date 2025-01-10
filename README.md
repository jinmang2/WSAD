# Weakly Supervised learning Video Anomaly Detection

## ENV

```shell
$ conda install -y -c pytorch -c nvidia pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1
$ conda install -y -c conda-forge pkg-config ffmpeg
$ conda install -y -c nvidia cuda=12.1 cuda-nvcc=12.1 cuda-toolkit=12.1
$ conda install -y -c "nvidia/label/cuda-12.1.0" \
    cuda-cccl cuda-compiler cuda-cudart-dev cuda-cudart-static cuda-cuobjdump cuda-cuxxfilt \
    cuda-demo-suite cuda-documentation cuda-driver-dev cuda-libraries-static \
    cuda-nsight cuda-nsight-compute cuda-nvml-dev cuda-nvprune \
    cuda-nvrtc-static cuda-nvvp cuda-opencl cuda-opencl-dev cuda-profiler-api cuda-sanitizer-api
$ conda install -y -c nvidia cuda-gdb=12.1 cuda-libraries-dev=12.1 cuda-nvdisasm=12.1 cuda-nvprof=12.1 cuda-version=12.1 cuda-visual-tools=12.1
$ conda install -y -c conda-forge gcc_linux-64 gxx_linux-64 libstdcxx-ng=12
$ pip install transformers tokenizers datasets huggingface_hub trl accelerate peft bitsandbytes deepspeed evaluate sentence-transformers sentencepiece "flash-attn>=2.5.8"
$ pip install einops matplotlib seaborn scikit-learn natsort openpyxl zipp
$ pip install hydra-core omegaconf lightning torchmetrics
$ pip install decord pytorchvideo torchcodec opencv-python segment-anything
$ pip install gradio streamlit wandb fastapi uvicorn rich packaging "PyYAML>=6.0"
$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
$ export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
$ export LD_PRELOAD=/lib/x86_64-linux-gnu/libtinfo.so.6
$ git clone --recursive https://github.com/dmlc/decord
$ cd decord
$ mkdir build && cd build
$ /opt/anaconda3/bin/cmake .. -DUSE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DFFMPEG_ROOT=$CONDA_PREFIX/bin/ffmpeg \
    -DCMAKE_C_COMPILER=$(which gcc) \
    -DCMAKE_CXX_COMPILER=$(which g++) \
    -DCUDA_NVIDIA_ML_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so
$ make
$ cd ../python
$ python setup.py install --user
```
```
# debug
$ ls -l $CONDA_PREFIX/lib/libstdc++.so.6
$ ldd /home/jinmang2/editable_packages/decord/build/libdecord.so | grep libstdc++.so.6
$ strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
```

- decord gpu test

```python
import datasets
from datasets import load_dataset
from decord import VideoReader, gpu, cpu

dataset = load_dataset("Wild-Heart/Disney-VideoGeneration-Dataset", split="train")
video_info = dataset.cast_column("video", datasets.Video(decode=False))[0]
video_path = video_info["video"]["path"]

vr_gpu = VideoReader(video_path, ctx=gpu(0))
frame = vr_gpu[0]
```

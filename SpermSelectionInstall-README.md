# Sperm Selection version2.0

## 1. install CUDA & cuDNN

### 1.1 install CUDA-12.1

- Download URL: `https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local`
- Add `YOUR_PATH\cuda12.1-install\libnvvp` to environmental path
- Add `YOUR_PATH\cuda12.1-install\bin` to environmental path

![](/readme-imgs/cuda-download.png)

### 1.2 install cuDNN-8.7.29

- Download URL: `https://developer.nvidia.com/rdp/cudnn-archive`
- Copy `bin\`, `include\` & `lib\` folders to CUDA install folder

![](/readme-imgs/cudnn-download.PNG)



## 2. conda virtual environment conduction

### 2.1 create virutal environmen
```
conda create -n spermselectionv2 python=3.9

# install pytorch
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# install onnx
pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

# install requestments.txt
pip install -r requestments.txt
```

### 2.2 install tensorrt
> CUDA 12.1, python=3.9

1. Download tensorrt tar file from: `https://developer.nvidia.com/nvidia-tensorrt-8x-download`

![](/readme-imgs/tensorrt-download.png)

2. install python wheel in `.YOUR_PATH\TensorRT-8.6.1.6\python\`

```
pip install tensorrt-8.6.1-cp39-none-win_amd64.whl
```

3. add tensorrt `lib` & `dll` files to environmental path

- copy `TensorRT-8.6.1.6\include` .h files to`CUDAv12.1-install\lib\include`；
- copy  `TensorRT-8.6.1.6\lib` .lib files to `CUDAv12.1-install\lib\x64`；
- copy `TensorRT-8.6.1.6\lib` .dll files to `CUDAv12.1-install\bin`；

4. test installation

```python
import onnxruntime as ort
import tensorrt
print(ort.get_device())
print(ort.get_available_providers())
print(tensorrt.__version__)
```

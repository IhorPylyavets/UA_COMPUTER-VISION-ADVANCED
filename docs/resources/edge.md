# Компресія, квантизація та edge-інференс: інструменти на GitHub

Заняття 10, 15–17, трек 4 фінального проєкту. Тут бібліотеки, які покривають багато методів або цілих рантаймів.
Стан — листопад 2026: перед використанням перевір дату останнього коміту.

Головне правило: квантизувати й міряти треба тим інструментом, що відповідає рантайму на цільовому залізі.
INT8 з одного стеку не переноситься в інший без втрат.

## Компресія та квантизація

| Репозиторій | Методи | Цільовий рантайм |
|---|---|---|
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | PTQ (INT8, FP8, INT4, NVFP4), QAT, прунінг, дистиляція, sparsity | TensorRT, TensorRT-LLM; колишній `TensorRT-Model-Optimizer` |
| [openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf) | PTQ, QAT, стиснення ваг, прунінг | OpenVINO, також ONNX і PyTorch |
| [intel/neural-compressor](https://github.com/intel/neural-compressor) | статична й динамічна PTQ, SmoothQuant, weight-only, QAT, mixed precision | CPU Intel, ONNX Runtime |
| [qualcomm/aimet](https://github.com/qualcomm/aimet) | PTQ (AdaRound, Cross-Layer Equalization, SeqMSE), QAT | Qualcomm SoC; колишній `quic/aimet` |
| [pytorch/ao](https://github.com/pytorch/ao) (torchao) | INT8 / INT4 / float8 квантизація, QAT, sparsity | Нативний PyTorch, `torch.compile` |
| [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning) | структурний прунінг на DepGraph: magnitude, Taylor, Hessian, BN-scale | Будь-який; приклади для YOLOv7/v8, ViT, SAM, LLM |

## Рантайми

| Репозиторій | Залізо | Примітка |
|---|---|---|
| [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) | NVIDIA GPU, Jetson | Найшвидший на Jetson; двигун будується під конкретний GPU і версію |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | CPU, CUDA, TensorRT, OpenVINO, CoreML, NNAPI та ін. як execution providers | Один формат — багато заліза |
| [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) | CPU, iGPU і NPU Intel, ARM CPU | Apache-2.0 |
| [Tencent/ncnn](https://github.com/Tencent/ncnn) | ARM CPU, Vulkan GPU | Мобільні й вбудовані плати без CUDA |
| [alibaba/MNN](https://github.com/alibaba/MNN) | ARM CPU, OpenCL, Vulkan, Metal | Apache-2.0 |

## Jetson і відеоконвеєри

| Репозиторій | Що всередині | Примітка |
|---|---|---|
| [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) | готові контейнери під JetPack: PyTorch, TensorRT, ONNX Runtime, VLM, ROS | Економить дні на збиранні залежностей |
| [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference) | класифікація, детекція, сегментація, поза на TensorRT + камера і RTP | Навчальний старт на Jetson |
| [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps) | приклади DeepStream на Python: декодування, інференс, трекер `nvtracker`, кілька потоків | Апаратний декодер і zero-copy із заняття 10 |

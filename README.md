# Progressive Diffusion-Guided Noise Perturbation for Data-Free Quantization
This repository is the official code for the paper "Progressive Diffusion-Guided Noise Perturbation for Data-Free Quantization" by Biao Qian, Yang Wang, Zeqian Yi, Haipeng Liu, Jungong Han and Meng Wang.


### Dependencies

Python 3.8
PyTorch 1.12.1


### Set the paths of datasets

Set the "dataPath" in "cifar100_resnet20.hocon" as the path root of your CIFAR100 dataset. For example:

        dataPath = "./dataset/CIFAR100/"


### Training

To quantize the pre-trained ResNet-20 on CIFAR100 to 4 bits:

    python main_direct.py --model_name resnet20_cifar100 --conf_path cifar100_resnet20.hocon --id=0


### Results

---------------------------------------------------------------------------------
| Dataset  |   Model   | Full-precision network  | Quantized network with 4 bits| 
--------------------------------------------------------------------------------- 
| CIFAR100 | ResNet-20 |         70.33%          |            66.09%            |
---------------------------------------------------------------------------------

Note that we use the pre-trained models (ResNet-20) from [pytorchcv](https://www.cnpython.com/pypi/pytorchcv).


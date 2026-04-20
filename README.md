# Progressive Diffusion-Guided Noise Perturbation for Data-Free Quantization
This repository is the official code for the paper "Progressive Diffusion-Guided Noise Perturbation for Data-Free Quantization" by Biao Qian, Yang Wang, Zeqian Yi, Haipeng Liu, Jungong Han and Meng Wang.


## Dependencies

Python 3.8
PyTorch 1.12.1


## Set the paths of datasets

For example, we can set the "dataPath" in "cifar100_resnet20.hocon" as the path root of your CIFAR100 dataset:

        dataPath = "./dataset/CIFAR100/"


## Training

For example, to quantize the pre-trained ResNet-20 on CIFAR100 to 4 bits, we can run:

    python main_direct.py --model_name resnet20_cifar100 --conf_path cifar100_resnet20.hocon --id=0


## Results

The performance of our models is measured by Top-1 classification accuracy (%), which is reported below:

![]()





Note that we use the pre-trained models (ResNet-20) from [pytorchcv](https://www.cnpython.com/pypi/pytorchcv).


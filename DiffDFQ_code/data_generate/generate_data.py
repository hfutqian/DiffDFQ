
import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *
import utils as utils

# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50','mobilenet_w1',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet20_cifar100', 'regnetx_600m'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--group',
                        type=int,
                        default=1,
                        help='group of generated data')
    parser.add_argument('--targetPro',
                        type=float,
                        default=1.0,
                        help='targetPro')
    parser.add_argument('--cosineMargin',
                        type=float,
                        default=0.4,
                        help='cosineMargin')
    parser.add_argument('--cosineMargin_upper',
                        type=float,
                        default=0.4,
                        help='cosineMargin_upper')
    parser.add_argument('--augMargin',
                        type=float,
                        default=0.4,
                        help='interClassMargin')
    parser.add_argument('--save_path_head',
                        type=str,
                        default='',
                        help='save_path_head')
    parser.add_argument('--qw',
                        type=int,
                        default=3,
                        help='the bit of quantization weight')
    parser.add_argument('--qa',
                        type=int,
                        default=3,
                        help='the bit of quantization activation')

    args = parser.parse_args()
    return args


def quantize_model(model, qw, qa):
        """
        Recursively quantize a pretrained single-precision model to int8 quantized model
        model: pretrained single-precision model
        """

        weight_bit = qw    #self.settings.qw
        act_bit = qa       #self.settings.qa

        # quantize convolutional and linear layers
        if type(model) == nn.Conv2d:
            quant_mod = Quant_Conv2d(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.Linear:
            quant_mod = Quant_Linear(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod

        # quantize all the activation
        elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
            return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])

        # recursively use the quantized module to replace the single-precision module
        elif type(model) == nn.Sequential:
            mods = []
            for n, m in model.named_children():
                mods.append(quantize_model(m,qw,qa))
            return nn.Sequential(*mods)
        else:
            q_model = copy.deepcopy(model)
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    setattr(q_model, attr, quantize_model(mod,qw,qa))
            return q_model


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    if args.model == 'regnetx_600m':
        from models.regnet import regnetx006
        model = regnetx006(pretrained=True)
    else:
        model = ptcv_get_model(args.model, pretrained=True)
        model_student = ptcv_get_model(args.model, pretrained=True)
    print('****** Full precision model loaded ******')

    # # Load validation data
    # test_loader = getTestData(args.dataset,
    #                           batch_size=args.test_batch_size,
    #                           path='/media/disk1/ImageNet2012/',
    #                           for_inception=args.model.startswith('inception'))
    # print('****** Test model! ******')
    # test(model.cuda(), test_loader)
    # Generate distilled data
    DD = DistillData(args.qw,args.qa)
    print(args.group, args.targetPro)

    model_student = quantize_model(model_student, args.qw, args.qa)


    dataloader = DD.getDistilData_hardsample_cosineDistanceEMA_interClass_aug(
        model_name=args.model,
        teacher_model=model.cuda(),
        student_model=model_student.cuda(),
        batch_size=args.batch_size,
        qw=args.qw,
        qa=args.qa,
        group=args.group,
        targetPro=args.targetPro,
        cosineMargin=args.cosineMargin,
        cosineMargin_upper=args.cosineMargin_upper,
        augMargin=args.augMargin,
        save_path_head=args.save_path_head
    )

    print('****** Data Generated ******')





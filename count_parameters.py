from config import config
from models.stacked_hourglass import PoseNet, SGSC_PoseNet
import torch
import sys
import gc
import time



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_res = config['train']['input_res']

    checkpoint_path = config['inference']['checkpoint_path']

    inp_dim = config['inference']['inp_dim']
    oup_dim = config['inference']['oup_dim']
    nstack = config['inference']['nstack']
    architecture = config['architecture']

    start_model = time.time()
    # default model is SGSC
    if architecture == 'SHG':
        net = PoseNet(nstack, inp_dim, oup_dim)
    else:
        net = SGSC_PoseNet(nstack, inp_dim, oup_dim)
    end_model = time.time()
    print('model loadeing time: ', (end_model - start_model)/60, 'minutes')

    start_cuda = time.time()
    net = net.cuda()
    end_cuda = time.time()
    print('model loading to cuda time: ', (end_cuda - start_cuda)/60, 'minutes')

    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    # for k, v in state_dict.items():
    #     print(k)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    number_params = count_parameters(net)
    print(f'[{architecture}] number of parameters: {number_params}')

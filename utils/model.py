import time
import torch
from models.stacked_hourglass import PoseNet, SGSC_PoseNet


def load_model(config):
    inp_dim = config['inference']['inp_dim']
    oup_dim = config['inference']['oup_dim']
    nstack = config['inference']['nstack']
    architecture = config['architecture']

    # default model is SGSC
    if architecture == 'SHG':
        net = PoseNet(nstack, inp_dim, oup_dim)
    else:
        net = SGSC_PoseNet(nstack, inp_dim, oup_dim)

    return net





def load_model_weights(config, net):
    checkpoint_path = config['inference']['checkpoint_path']
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

from config import config
from data.MPII.dataset import MPII_ValidationDataset
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader
import gc
import time
from PIL import Image, ImageDraw
import torchvision
import copy
from utils.keypoints import get_keyppoints, post_process_keypoints, get_keyppoints_without_adjustment
from utils.img import save_image
import utils.model


# gt: ground truth
def mpii_eval(pred, gt, normalizing, bound=0.5):
    """
    Use PCK with threshold of .5 of normalized distance (presumably head size)
    """

    correct = {'all': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
               'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
               'shoulder': 0},
               'visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
               'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
               'shoulder': 0},
               'not visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
               'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
               'shoulder': 0}}
    count = copy.deepcopy(correct)
    idx = 0
    for p, g, normalize in zip(pred, gt, normalizing):

        for j in range(g.shape[1]):
            vis = 'visible'

            if g[0,j,0].item() == 0: ## not in picture!
                continue
            if g[0,j,2].item() == 0:
                vis = 'not visible'
            joint = 'ankle'
            if j==1 or j==4:
                joint = 'knee'
            elif j==2 or j==3:
                joint = 'hip'
            elif j==6:
                joint = 'pelvis'
            elif j==7:
                joint = 'thorax'
            elif j==8:
                joint = 'neck'
            elif j==9:
                joint = 'head'
            elif j==10 or j==15:
                joint = 'wrist'
            elif j==11 or j==14:
                joint = 'elbow'
            elif j==12 or j==13:
                joint = 'shoulder'

            count['all']['total'] += 1
            count['all'][joint] += 1
            count[vis]['total'] += 1
            count[vis][joint] += 1

            error = np.linalg.norm(p[0, j, :2]-g[0,j,:2]) / normalize # L2 norm

            if bound > error:
                correct['all']['total'] += 1
                correct['all'][joint] += 1
                correct[vis]['total'] += 1
                correct[vis][joint] += 1

        idx += 1

    for k in correct:
        print(k, ':')
        for key in correct[k]:
            print('Val PCK @,', bound, ',', key, ':', round(correct[k][key] / max(count[k][key],1), 3), ', count:', count[k][key])
        print('\n')


def inference(model, criterion, valid_ds):
    valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        outputs_valid = []
        targets_valid = []
        normalizing = []
        gts = [] # ground truths
        preds = [] # predictions
        valid_losses_list = []
        for j, data in enumerate(valid_dataloader):
            input_img = data['image'] # the image is preprocessed for inference (cropped, resized and normalized)
            heatmaps = data['heatmaps'] # shape [bs, 16, 64, 64]
            n = data['normalize'].item()
            normalizing.append(n)
            orig_keypoints = data['orig_keypoints'] # (1, 16, 3)
            gts.append(orig_keypoints.numpy())
            c = (data['center'][0, 0].item(), data['center'][0, 1].item())
            s = data['scale'].item()
            input_res = data['input_res']

            inputs = input_img.cuda()

            start_prediction = time.time()
            preds_valid = model(inputs) # shape (1, 4, 16, 64, 64) = (bs, hg_modules, 16 kp, height, width)
            end_prediction = time.time()
            print('model prediction time: ', (end_prediction - start_prediction), 'seconds')
            preds_valid = preds_valid.cpu()

            valid_loss = criterion(combined_hm_preds=preds_valid, heatmaps=heatmaps) # loss shape = (16, 8)
            valid_loss = torch.mean(valid_loss)
            valid_losses_list.append(valid_loss.item())
            outputs_valid.append(preds_valid)
            targets_valid.append(heatmaps)

            # get keypoints from predicted heatmaps as (x, y) = (width, height)
            pred_keypoints = get_keyppoints(preds_valid[:, -1]) # returns (batch_size, 16, 2)

            keypoints = post_process_keypoints(pred_keypoints, input_img, c, s, input_res)
            preds.append(keypoints)
        outputs_valid = torch.cat(outputs_valid)
        targets_valid = torch.cat(targets_valid)
    overall_valid_loss = sum(valid_losses_list)/len(valid_losses_list)
    print(f'-- VALIDATION LOSS: ', overall_valid_loss)
    return preds, gts, normalizing


if __name__ == '__main__':
    # set up hyperparameters
    bs = config['train']['batchsize']
    lr = config['train']['learning_rate']
    input_res = config['train']['input_res']
    epochs = config['train']['epochs']
    checkpoint_path = config['inference']['checkpoint_path']

    valid_ds = MPII_ValidationDataset(config, mode='valid')
    torch.cuda.empty_cache()
    gc.collect()
    net = utils.model.load_model(config)

    # model loading to cuda time:  0.8921654939651489 minutes
    start_cuda = time.time()
    net = net.cuda()
    end_cuda = time.time()
    print('model loading to cuda time: ', (end_cuda - start_cuda)/60, 'minutes')
    criterion = net.calc_loss
    net = utils.model.load_model_weights(config, net)
    net.eval()

    # preds list with elements of shape (1, 16, 2)
    # gts list with elements of shape = (1, 16, 3)
    preds, gts, normalizing = inference(net, criterion, valid_ds)
    mpii_eval(preds, gts, normalizing)

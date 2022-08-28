from config import config
from data.MPII.load_data import MPII
from data.MPII.dataset import MPII_ValidationDataset
from utils.keypoints import get_keyppoints, post_process_keypoints, draw_cv2_keypoints
from utils.img import save_image
import torch
import sys
import time
import utils.img
import utils.model


start_model = time.time()
net = utils.model.load_model(config)
end_model = time.time()
print('model loadeing time: ', (end_model - start_model)/60, 'minutes')

start_cuda = time.time()
net = net.cuda()
end_cuda = time.time()
print('model loading to cuda time: ', (end_cuda - start_cuda)/60, 'minutes')

net = utils.model.load_model_weights(config, net)
net.eval()

mpii = MPII()
# validation set has 2958 images
idx = 580
dataset = MPII_ValidationDataset(config, mode='valid')
idx = dataset.index[idx % len(dataset.index)]
img_dict = dataset.loadImage(idx)

input_img = img_dict['image'].unsqueeze(dim=0) # the image is already preprocessed for inference (cropped and normalized)
heatmaps = img_dict['heatmaps'].unsqueeze(dim=0) # shape [1, 16, 64, 64]
n = img_dict['normalize'].item()
orig_keypoints = torch.from_numpy(img_dict['orig_keypoints']).unsqueeze(dim=0) # (1, 16, 3)
c = (img_dict['center'][0].item(), img_dict['center'][1].item())
s = img_dict['scale'].item()
input_res = img_dict['input_res']
orig_img = img_dict['orig_img']
orig_img_tmp = orig_img.clone()
orig_img_cv2 = orig_img.permute(1, 2, 0).numpy() # convert to cv2 image

inputs = input_img.cuda()

start_prediction = time.time()
with torch.no_grad():
    preds = net(inputs) # shape (1, 4, 16, 64, 64) = (bs, hg_modules, 16 kp, height, width)
end_prediction = time.time()
print('model prediction time: ', (end_prediction - start_prediction), 'seconds')
preds = preds.cpu()

pred_keypoints = get_keyppoints(preds[:, -1]) # returns (batch_size, 16, 2)
keypoints = post_process_keypoints(pred_keypoints, input_img, c, s, input_res)
img_with_keypoints = draw_cv2_keypoints(orig_img_cv2, keypoints[0], radius=9)

dir = config['inference']['presentation_dir']

## save as image: ##

# original image
save_image(orig_img_tmp, dir+'original_image.png')

# preprocessed image
save_image(input_img[0], dir+'preprocessed_image.png')

# target heatmap with head keypoint
save_image(heatmaps[0, 9], dir+'target_heatmap.png', permute=False)

# predicted heatmap
save_image(preds[0, -1,  9], dir+'predicted_heatmap.png', permute=False)

# image with skeleton
save_image(img_with_keypoints, dir+'img_with_keypoints.png', permute=False)

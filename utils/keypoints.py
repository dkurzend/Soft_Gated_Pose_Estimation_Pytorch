import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import utils.img
import sys
import cv2


# input shape: (batch_size, 16, 64, 64)
def get_keyppoints(heatmaps):
    batch_size = heatmaps.shape[0]
    num_keypoints = heatmaps.shape[1]
    keypoints = np.zeros((batch_size, num_keypoints, 2)) # shape (bs, 16, 2)

    for batch in range(batch_size):
        for kp in range(num_keypoints):
            hm = heatmaps[batch, kp, :,:]
            # flatten the thensor and get indices of 2 highest values
            v, i = torch.topk(hm.flatten(), 2)

            # convert back to 2d matrix
            # keypoint[0] contains height or y axis
            # keypoint[1] contains width or x axis
            top_k = np.array(np.unravel_index(i.numpy(), hm.shape)).T # top_K has shape [2, 2]
            keypoint = torch.tensor(top_k[0]).type(torch.FloatTensor)

            # move y one quarter pixel towards next highest pixel
            if top_k[1, 0] < keypoint[0]:
                keypoint[0] -= 0.25
            if top_k[1, 0] > keypoint[0]:
                keypoint[0] += 0.25

            # move x one quarter pixel towards next highest pixel
            if top_k[1, 1] < keypoint[1]:
                keypoint[1] -= 0.25
            if top_k[1, 1] > keypoint[1]:
                keypoint[1] += 0.25

            # we return the keypoints as (x, y) = (width, height) coordinates
            keypoints[batch, kp, 0] = keypoint[1]
            keypoints[batch, kp, 1] = keypoint[0]

    return keypoints


# input shape: (batch_size, 16, 64, 64)
def get_keyppoints_without_adjustment(heatmaps):
    batch_size = heatmaps.shape[0]
    num_keypoints = heatmaps.shape[1]
    keypoints = np.zeros((batch_size, num_keypoints, 2)) # shape (16, 16, 2)

    for batch in range(batch_size):
        for kp in range(num_keypoints):
            hm = heatmaps[batch, kp, :,:]

            # keypoint[0] contains height or y axis
            # keypoint[1] contains width or x axis
            keypoint = (hm==torch.max(hm)).nonzero()[0]

            # we return the keypoints as (x, y) = (width, height) coordinates
            keypoints[batch, kp, 0] = keypoint[1]
            keypoints[batch, kp, 1] = keypoint[0]

    return keypoints




# keypoint: (batch_size, 16, 2)
# img: [1, 3, 256, 256]
def post_process_keypoints(keypoints, img, c, s, input_res):
    height, width = img.shape[2:]
    center = (width/2, height/2)
    scale = max(height, width)/200
    res = (input_res, input_res)

    mat_ = utils.img.get_transform(center, scale, res)[:2]
    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2]

    # transform to 256x256 resolution
    keypoints[:,:,:2] = utils.img.kpt_affine(keypoints[:,:,:2] * 4, mat)
    preds = np.copy(keypoints)

    # transform each keypoint to orig resolution
    for j in range(preds.shape[1]):
        preds[0,j,:2] = utils.img.transform(preds[0,j,:2], c, s, res, invert=1)
    return preds # shape: [1, 16, 2]



# keypoints shape = (16, 2)
# kp[0] = x axis = width
# kp[1] = y axis = height
# image is a cv2 image
def draw_cv2_keypoints(image, keypoints, radius=4):
    num_keypoints = keypoints.shape[0]

    # draw lines
    for i in range(num_keypoints):
        draw_cv2_lines_between_keypoints(image, keypoints)

    # draw points
    for i in range(num_keypoints):
        if (keypoints[i,0]>=1) and (keypoints[i,1]>=1):
            point = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            image = cv2.circle(image, point, radius=radius, color=(255, 0, 0), thickness=-1)

    return image



# draw lines between keypoints
def draw_cv2_lines_between_keypoints(image, keypoints):
    pairs = [[8,9],[7,8], [7,12],[7,13], [11,12],[13,14], [10,11],[14,15], [6,7],[2,6], [3,6],[1,2], [3,4],[0,1], [4,5]]
    for list in pairs:
        if (keypoints[list[0],0]>=1) and (keypoints[list[0],1]>=1) and (keypoints[list[1],0]>=1) and (keypoints[list[1],1]>=1):
            point = (int(keypoints[list[0], 0]), int(keypoints[list[0], 1]))
            point2 = (int(keypoints[list[1], 0]), int(keypoints[list[1], 1]))
            image = cv2.line(image, point,point2,(0,255,0), 3)






# keypoints shape = (16, 2)
# kp[0] = x axis = width
# kp[1] = y axis = height
def draw_PIL_keypoints(image, keypoints, radius=12):
    transform_to_pillow = transforms.ToPILImage()
    image = transform_to_pillow(image)
    draw = ImageDraw.Draw(image)
    num_keypoints = keypoints.shape[0]

    for i in range(num_keypoints):

        draw.ellipse((keypoints[i, 0]-radius, keypoints[i, 1]-radius, keypoints[i, 0]+radius, keypoints[i, 1]+radius), fill = 'red', outline ='green')

    convert_to_tensor = transforms.ToTensor()
    image = convert_to_tensor(image) # returns shape (C x H x W)
    return image

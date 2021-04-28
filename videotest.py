# @Time : 2021-04-22 14:25
# @Author : Xuanhan Liu
# @Site :
# @File : test2.py
# @Software: PyCharm
import skvideo.io
from cv2 import os
import cv2
import av
import transformer_net
import torch
import numpy as np
import cv2.aruco as aruco
from config import config
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import tqdm
import os

style_model = transformer_net.TransformerNet()
style_model.load_state_dict(torch.load(config.model_path, map_location=lambda _s, _: _s))
style_model = style_model.cuda()

def findArucoMakers(img, maker_size=6, total_maker=100, draw=True):
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    getattr(aruco, f'DICT_4X4_50')
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParam = aruco.DetectorParameters_create()
    box, ids, rejected = aruco.detectMarkers(imGray, arucoDict, parameters=arucoParam)
    print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, box)
    return [box, ids]


def augmentAruco(box, id, img, imgaug, drawId=True):
    tl = box[0][0][0], box[0][0][1]
    tr = box[0][1][0], box[0][1][1]
    br = box[0][2][0], box[0][2][1]
    bl = box[0][3][0], box[0][3][1]
    h, w, c = img.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgaug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgOut = img + imgOut
    return imgOut

def extract_frames(video_path):
    """ Extracts frames from video """
    video = av.open(video_path)
    for frame in video.decode(0):

        yield frame.to_image()


with torch.no_grad():
    cap = cv2.VideoCapture(0)
    i = 0
    while cap.isOpened():
        imgAug = cv2.imread('pics/b6d4305ff2b5f2de5833054853f3deb5.jpeg')
        imgAug = cv2.resize(imgAug, (2000, 2000))
        ret, frame = cap.read()
        aruco_find = findArucoMakers(frame)
        if len(aruco_find[0]) != 0:
            for box, id in zip(aruco_find[0], aruco_find[1]):
                frame = augmentAruco(box, id, frame, imgAug)
        i += 1
        b, g, r = cv2.split(frame)
        frame = cv2.merge([r, g, b])

        frame = torch.Tensor(frame).permute(2, 0, 1)
        frame = frame.unsqueeze(0).cuda()
        output = style_model(frame)
        output = output.cpu().data[0].squeeze(0).permute(1, 2, 0).numpy()
        output = output/255.
        r, g, b = cv2.split(output)
        output = cv2.merge([b, g, r])
        cv2.imshow('realtime',output)
        cv2.waitKey(1)
        cv2.imwrite('output3/000000{}.jpg'.format(i), output*255.)




# writer = skvideo.io.FFmpegWriter()
# frames = []
# images = extract_frames('IMG_0714.MOV')
# with torch.no_grad():
#     for ii, image in tqdm.tqdm(enumerate(images), total=430, desc="Processing frames"):
#         x = Variable(torch.Tensor(np.array(image)))
#         x = x.permute(2, 0, 1).unsqueeze(0)
#         x = x.cuda()
#         output = style_model(x).cpu().data[0].squeeze(0)
#         output_data = output.permute(1, 2, 0).numpy().astype(np.uint8)
#         frames += [output_data]
#
#
# for frame in tqdm.tqdm(frames, total=430, desc="Writing to video"):
#     writer.writeFrame(frame)
# writer.close()

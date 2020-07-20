
from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2
from PIL import Image
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
import matplotlib.pyplot as plt
import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from helpers.visualize_yoga import cv_plot_keypoints
import cv2
import time
import argparse
import imutils
from imutils.video import VideoStream
import multiprocessing as mp
from multiprocessing import Pool
import os
import pandas as pd

ctx = mx.cpu()
#detector_name = 'yolo3_mobilenet1.0_coco'
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)
detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimator.hybridize()

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--vid', type = str, required = False, help ='video path')
ap.add_argument('-o', '--outfile', default = 'output.avi', help = 'outfile path')
args = vars(ap.parse_args())

num_cores = mp.cpu_count()
using_vid_file = False

if args['vid'] is not None:
    using_vid_file = True
    print('[INFO] using video file...')
    vid_path = os.getcwd() + args['vid'][1:]
    vs = cv2.VideoCapture(vid_path)
    
else:
    print('[INFO] using camera...')
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

#frame = vs.read()
#if using_vid_file:
#    frame = frame[1]
#(H, W) = frame.shape[:2]
vid_writer = cv2.VideoWriter(args['outfile'],cv2.VideoWriter_fourcc(*'MJPG'), 29, ( 500, 280), True)

def main():
    start = time.time()
    count = 0
    pose = None
    skip_frame = False

    #with open('data.txt', 'a') as outfile:
    
    while True:

        frame = vs.read()
        if using_vid_file:
            frame = frame[1]
        try:
            frame = np.fliplr(frame)
        except ValueError:
            print('[ERROR] video file not found, make sure to include path and extension i.e. \'./vid.mp4\'')
            break
        count += 1
        frame = imutils.resize(frame, width=280)  
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        if not skip_frame:
            x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short = 512, max_size = 280)
            x = x.as_in_context(ctx)

            class_IDs, scores, bounding_boxs = detector(x)

            pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                               output_shape=(128, 96), ctx=ctx)
            if len(upscale_bbox) > 0:
                predicted_heatmap = estimator(pose_input)
                pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
                img, pose = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, None, scores,
                                        box_thresh=0.5, keypoint_thresh=0.15)

                #The following lines were for saving vid info to a file
                #pose = args['vid'].split('/')
                #pose = pose[2][:-4]
                #outfile.write(pose + ', ')
                #for angle in angles:
                #    outfile.write(str(angle) + ', ')
                #outfile.write('\n')
                

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            skip_frame = True

        else:
            skip_frame = False
        img = imutils.resize(img, height = 280, width = 500)
        vid_writer.write(img)
        if pose:
            cv2.putText(img, '{}'.format(pose), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'No Pose Detected', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    if not using_vid_file:
        vs.stop()
    vid_writer.release()
    stop = time.time()
    #outfile.close()
    print("fps: {}".format(count/(stop-start)))
    

if __name__ == '__main__': 
    
    main()
   


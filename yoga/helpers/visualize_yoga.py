from __future__ import absolute_import, division

import mxnet as mx
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import os

np.seterr(divide='ignore', invalid='ignore')

model_path = os.getcwd() + '\\helpers\\model' # get path of saved Keras Model
model = load_model(model_path)                # load model

def predict_pose(angles):
    """
    Takes a list of list of angles made by the joints of the body in a particular frame.
    Runs angles through model to predict pose
    returns string: confidence of prediction and name of pose
    """
    POSE_THRESHOLD = 0.8

    angles = np.array([(angles)])
    labels = {0:'Bound Angle Pose', 
              1: 'Cat Cow', 
              2: 'Chair Pose', 
              3: 'Cobra', 
              4: 'Downward Dog', 
              5: 'Forward Bend', 
              6: 'Garland', 
              7: 'Half Forward Bend', 
              8: 'High Lunge', 
              9: 'Plank', 
              10: 'Reverse Warrior', 
              11: 'Seated Spinal Twist', 
              12: '3 Leg Downward Dog', 
              13: 'Tree', 
              14: 'Triangle', }
    scores = model.predict(angles)
    max_elm = np.amax(scores[0])
    result = np.where(scores[0] == max_elm)
    if result[0][0]:
        result = result[0][0]
        if max_elm > POSE_THRESHOLD:
            #print('{:.1f}% {}'.format(max_elm*100, labels[result]))
            return '{:.1f}% {}'.format(max_elm*100, labels[result])
    else:
        return None

def cv_plot_keypoints(img, coords, confidence, class_ids, bboxes, scores,
                      box_thresh=0.5, keypoint_thresh=0.2, scale=1.0, **kwargs):
    """Visualize keypoints with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    coords : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 2`.
    confidence : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 1`.
    class_ids : numpy.ndarray or mxnet.nd.NDArray
        Class IDs.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    box_thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `box_thresh`
        will be ignored in display.
    keypoint_thresh : float, optional, default 0.2
        Keypoints with confidence less than `keypoint_thresh` will be ignored in display.
    scale : float
        The scale of output image, which may affect the positions of boxes

    Returns
    -------
    numpy.ndarray
        The image with estimated pose.
    pose
        string of with pose name and confidence for displaying

    """

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    if isinstance(class_ids, mx.nd.NDArray):
        class_ids = class_ids.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if isinstance(confidence, mx.nd.NDArray):
        confidence = confidence.asnumpy()
    joint_pairs_with_eyes = [[0, 1], [1, 3], [0, 2], [2, 4],  # use this dictionary if you want to track face features
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]
    joint_pairs = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # use this dictionary to not include face features
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]
    pose = None
    angles = []
    joint_visible = confidence[:, :, 0] > keypoint_thresh   # an array of boolean values, true if point is above confidence thresh

    coords *= scale                          # coords were scaled so we have to rescale here
    for i in range(coords.shape[0]):         # In this case, i is always one, only processing a single frame at a time
        pts = coords[i]
        GREEN = (0,255,0)
        RED = (255, 0, 0)
        for jp in joint_pairs:
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:    # if both points above confidence threshold, display a line between them
                pt1 = (int(pts[jp, 0][0]), int(pts[jp, 1][0]))
                pt2 = (int(pts[jp, 0][1]), int(pts[jp, 1][1]))
                cv2.line(img, pt1, pt2, GREEN, 3)
        

    def findAngle(A, cent, B, pt1, pt2, pt3):
        """ nested function for determining the angle (in radians) between joints
        input:
            A, cent, B : coords of the three joints
            pt1, pt2, pt3:  the label of the three joints
        output: 
            angle in radians
        """
        if joint_visible[i, pt1] and joint_visible[i, pt2] and joint_visible[i, pt3]:
            a1, a2 = A
            b1, b2 = B
            c1, c2 = cent
            a = (c1 - a1, c2 - a2)
            b = (c1 - b1, c2 - b2)
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        else:
            return None

    angles += [findAngle(pts[7], pts[5], pts[11], 7, 5, 11),  # shoulder
                   findAngle(pts[8], pts[6], pts[12], 8, 6, 12),  # shoulder
                   findAngle(pts[5], pts[7], pts[9], 5, 7, 9),   # elbow
                   findAngle(pts[6], pts[8], pts[10], 6, 8, 10),  # elbow
                   findAngle(pts[5], pts[11], pts[13], 5, 11, 13), # hip
                   findAngle(pts[6], pts[12], pts[14], 6, 12, 14), # hip
                   findAngle(pts[11], pts[13], pts[15], 11, 13, 15),# knee
                   findAngle(pts[12], pts[14], pts[15], 12, 14, 15),# knee
                    ]
    if all(angles):
        pose = predict_pose(angles)

          
    return img, pose








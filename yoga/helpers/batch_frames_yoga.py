import cv2
import numpy as np
def batch_frames(path, num_cores):
    video = cv2.VideoCapture(path)
    (grabbed, frame) = video.read()
    shape = frame.shape[:2]
    bank = np.array(frame)
    bank = np.expand_dims(bank, axis=0)
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()
        # check to see if we have reached the end of the video
        if grabbed:
            bank = np.append(bank, [frame], axis=0)
        else: 
            break
            
    video.release()
    chunk_size = len(bank) // num_cores + num_cores
    num_chunks = len(bank) // chunk_size
    chunks = []
    for i in range(num_chunks+1):
        chunks.append(bank[0:chunk_size,:,:,:])
    return chunks, bank, shape
    
    

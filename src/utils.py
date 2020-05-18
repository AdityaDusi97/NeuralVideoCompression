import numpy as np
import cv2
import os
import pdb

# TODO: Some random prints. Need to fix that


def Video2Frames(readfile: str, savefile: str) -> None:
    video = cv2.VideoCapture(readfile)
    status = True
    count = 0
    while status:
        status, frame = video.read()
        frame = np.transpose(frame, (1, 0, -1))
        saveName = os.path.join(savefile, "frame" + str(count) + ".png")
        cv2.imwrite(saveName, frame)
        count += 1
    print("Wrote Frames\n")


def Video2Residual(readpath: str, savePath: str, quality: int = 1) -> None:
    """
    params: 
    	readpath: path to video
    	savePath: path to save video
    	qualityL 0-100, 100 being best quality
    """
    video = cv2.VideoCapture(readpath)
    status = True
    count = 0
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        print('Creating output directory')
    else:
        print('Rewriting output directory')
    filePath = os.path.join(savePath, 'log.txt')
    file = open(filePath, 'w+')
    file.write("Source Video: " + readpath + "\n")
    file.write("Quality: %d" %(quality))
    file.close()
    while status:
        #pdb.set_trace()
        status, frame = video.read()
        encoded, encodedFrame = cv2.imencode('.jpg', frame, [quality, 0])
        decodedFrame = cv2.imdecode(encodedFrame, cv2.IMREAD_COLOR)
        saveName = os.path.join(savePath, "Frame" + str(count) + '.png')
        residual = frame - decodedFrame
        count +=1
        # format in in BGR
        cv2.imwrite(saveName, residual)
    print('Wrote Frames')





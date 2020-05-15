# All utility functions here
import numpy as np
import cv2
import os

#TODO: Some random prints. Need to fix that
def Video2Frames(readfile: str, savefile: str) -> None:
	video = cv2.VideoCapture(readfile)
	status = True
	count =0
	while status:
		status, frame = video.read()
		frame = np.transpose(frame, (1,0,-1))
		saveName = os.path.join(savefile, "frame" + str(count) + ".png")
		cv2.imwrite(saveName, frame)
		count +=1
	print("Wrote Frames\n")




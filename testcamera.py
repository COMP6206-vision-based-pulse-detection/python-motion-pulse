#!/usr/local/bin/python
# coding: utf-8

import cv2
import time
from memory_profiler import profile

@profile
def func():
	camera = cv2.VideoCapture("face2.mp4")

	for i in range(100):
		
			
		tt = time.time()
		retval = camera.grab()
		if (i+4)%2 == 0:
			go, capture = camera.retrieve(retval)
		print time.time() - tt
		#cv2.waitKey(1)
		#del capture

if __name__ == '__main__':
	func()
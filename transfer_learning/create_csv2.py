import csv
import os
import cv2
import time
import math

directory = 'images/train'
files = os.listdir(directory)

params = cv2.SimpleBlobDetector_Params()

# threshold
params.minThreshold = 0;
params.maxThreshold = 255;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 2
params.maxArea = 220
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.5
 
# Filter by Inertia
params.filterByInertia =False
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)

with open('data/train.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['filename'] + ['width']+['height']+['class']+['xmin']+['ymin']+['xmax']+['ymax'])
	for f in files:
		name = os.path.splitext(f)[0]
		stats = name.split('_')
		x = int(stats[4])
		y = int(stats[5])
		width = int(stats[2])
		height = int(stats[3])

		ball_radius = 20
		diameter = ball_radius
		xmin =max(0,x-ball_radius) 
		ymin =max(0,y-ball_radius)

		xmax = min(width,x+ball_radius)
		ymax = min(height,y+ball_radius)

		frame = cv2.imread('images/train/'+f)
		r,g,b = cv2.split(frame)

		# Blur image to remove noise
		g=cv2.GaussianBlur(g, (3, 3), 0)

		# Sets pixels to white if in purple range, else will be set to black
		g = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,1)

		# dilate makes the in range areas larger
		strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
		mask = cv2.dilate(g, strel, iterations=2)
		mask = cv2.erode(mask, strel, iterations=2)

		keypoints = detector.detect(mask[ymin:ymax,xmin:xmax])
		
		for point in keypoints:
			diameter = point.size

		ball_radius = int(diameter)
		xmin =max(0,x-ball_radius) 
		ymin =max(0,y-ball_radius)

		xmax = min(width,x+ball_radius)
		ymax = min(height,y+ball_radius)

		if stats[6]=='1':
				_class = 'ball'
				writer.writerow([f,width,height,_class,xmin,ymin,xmax,ymax])


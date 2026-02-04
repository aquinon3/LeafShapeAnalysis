import os, glob
import sys
import numpy as np
import cv2 as cv
import imutils
import pandas as pd
import matplotlib.pyplot as plt

impath = "/Users/quino070/LeafShapeAnalysis/images/" 

results_path = "/Users/quino070/LeafShapeAnalysis/V3test/output/LeafAnalysisResults_20251226.csv"

results = pd.read_csv(results_path)

results["leaf_number"] = results["leaf_number"].astype("Int64") 

NAleaves = results["leaf_number"].isna()
NAleaves = results.index[NAleaves].tolist()


for leaf in NAleaves:
	fullimage = results.iloc[leaf]["image"]
	obj = results.loc[leaf,"object_ID"]
	im = cv.imread(f'{impath}{fullimage}.tif')

	#Make grayscale to create binary mask
	gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

	#Make binary mask to identify objects
	_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+ cv.THRESH_OTSU)
	thresh = cv.erode(thresh, (5,5), iterations=3)    
    
    #identify  contours
	contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours) 
    
    #filter out objects that are not leaves
	min_area = 250000   # Minimum area threshold
	max_area = 10000000   # Maximum area threshold
	filtered_contours = [cnt for cnt in contours if 
		min_area < cv.contourArea(cnt) < max_area]

	sorted_contours= sorted(filtered_contours, key=cv.contourArea, reverse= True)

	contour = sorted_contours[obj-1]

	x_leaf, y_leaf, w_leaf, h_leaf = cv.boundingRect(contour)

	leaf_im = im[y_leaf:y_leaf+h_leaf, x_leaf:x_leaf+w_leaf]

	plt.imshow(leaf_im)
	plt.title(fullimage)
	plt.axis("off")
	plt.show(block=False)

	lab = input("manually added label: ")	

	plt.close()

	if lab=="NA":
		print("No change in label")
	else: 
		results.loc[results.index[leaf],"leaf_number"] = int(lab)

root_filepath= results_path.split("/")[-1]

results.to_csv(f"/Users/quino070/LeafShapeAnalysis/V3test/output/AfterManualAnn_{root_filepath}", index=False)










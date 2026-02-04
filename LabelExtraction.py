import os, glob
import sys
import numpy as np
import cv2 as cv
import imutils

        
def main():
    impath= sys.argv[1]
    impath = os.path.abspath(impath)
    outpath= "/Users/quino070/LeafShapeAnalysis/handwrittenlabels/images"
    outpath_labs = "/Users/quino070/LeafShapeAnalysis/handwrittenlabels/labels"
    ext="tif"
    files = glob.glob(os.path.join(impath, f'*.{ext}'))

    print("Extracting labels from images in "+ impath)
    
    for idx, im_path in enumerate(files):
        im = cv.imread(im_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.copyMakeBorder(im, 100, 100, 100, 100,
                                    cv.BORDER_CONSTANT, value=(255, 255, 255))
        
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+ cv.THRESH_OTSU)
        thresh = cv.erode(thresh, (5,5), iterations=3)
        
        contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        min_area = 250000   # Minimum area threshold
        max_area = 10000000   # Maximum area threshold
        filtered_contours = [cnt for cnt in contours if 
                             min_area < cv.contourArea(cnt) < max_area]
        sorted_contours= sorted(filtered_contours, key=cv.contourArea, reverse= True)
        
        small_min_area = 50
        small_max_area = 5000
        small_contours  = [cnt for cnt in contours if 
                           small_min_area < cv.contourArea(cnt) < small_max_area]
        
        cnt_mask = np.zeros_like(im)
        d0 =100
        cv.drawContours(cnt_mask, small_contours, -1, (255, 255, 255), d0)
        for cnt in small_contours:
            M = cv.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            d=50
            cv.circle(cnt_mask, (cx, cy), d, (255, 255, 255), -1) 
            
        im_og = im.copy()
        
        for c_id, contour in enumerate(sorted_contours):
            x_leaf, y_leaf, w_leaf, h_leaf = cv.boundingRect(contour)
            
            label = cv.cvtColor(cnt_mask[y_leaf:y_leaf+h_leaf, x_leaf:x_leaf+w_leaf], cv.COLOR_BGR2GRAY)
            tag_contour = cv.findContours(label, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            tag_contour = imutils.grab_contours(tag_contour)
            
            try:

                tag_contour= max(tag_contour, key = cv.contourArea)
                
                x_tag,y_tag,w_tag,h_tag = cv.boundingRect(tag_contour)
                x_tag = x_tag + x_leaf-100
                y_tag = y_tag + y_leaf
                w_tag = w_tag + 80
                h_tag = h_tag + 40
                #cv.rectangle(im_og,(x_tag,y_tag),(x_tag+w_tag,y_tag+h_tag),(255,0,0),20)
                
                tag_im = im_og[y_tag:y_tag+h_tag, x_tag:x_tag+w_tag]
                tag_im = cv.resize(tag_im, (300,300), interpolation=cv.INTER_CUBIC)
                tag_im = cv.cvtColor(tag_im, cv.COLOR_BGR2GRAY)
                
                cv.imwrite(f'{outpath}/{str(idx)}_{str(c_id)}.jpg', tag_im)

                im_path = im_path.replace(f'{impath}/', "").replace(".tif","")


                lab_file = f'{outpath_labs}/{str(idx)}_{str(c_id)}_coords.txt'
                lab_id = f'{str(idx)}_{str(c_id)}'
                with open(lab_file, "w") as file:
                    file.write("file,lab,xmin,ymin,xmax,ymax\n")
                    file.write(f'{im_path}, {lab_id}, {x_tag}, {y_tag}, {(x_tag+w_tag)}, {(y_tag+h_tag)}')




            
            except ValueError:
                print(f'No tag contour found in image {im_path}, leaf {c_id}')
                continue

    print("Done extracting labels. Images in " + outpath)

if __name__ == "__main__":
    main()
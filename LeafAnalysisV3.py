import os, glob
import sys
import numpy as np
import cv2 as cv
import imutils
import pandas as pd
from datetime import date
import argparse

date = date.today().strftime("%Y%m%d")
tag_dict = pd.read_csv("/Users/quino070/LeafShapeAnalysis/handwrittenlabels/LabelDict.csv")

# class options:
#     def __init__(self):
#         # Input image path/filename
#         self.image = "."
#         # Results path/filename
#         self.
#         # Image output directory path
#         self.outdir = "img"
        
def main():
    # Initialize options
    # args = options()
    # folder,save= input("enter foldername directory and output directory: ").split()
    # input_folder=os.path.abspath(folder)
    # output_folder=os.path.abspath(save)

    result = f'LeafAnalysisResults_{date}.csv'

    parser = argparse.ArgumentParser(description="Enter input and output directory.")
    parser.add_argument("indir", type=str, help="Input directory with no final /")
    parser.add_argument("outdir", type=str, help="Output directory with no final /")
    args = parser.parse_args()
    input_folder = os.path.abspath(args.indir)
    output_folder= os.path.abspath(args.outdir)



    ext="tif"
    files = glob.glob(os.path.join(input_folder, f'*.{ext}'))
    
    df = pd.DataFrame()

    for im in files:
        
        image_name = im.replace(f'{input_folder}/', "").replace(".tif","")

        #image tags
        file_tags = tag_dict[tag_dict["file"]==image_name]

        print(f"Processing: {image_name}")
        
        #args.image = input_folder+i
        args.image = im
        
        #Read in image
        image = cv.imread(args.image)
        #plt.imshow(image)
        
        #Make grayscale to create binary mask
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        
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
        
        #Draw leaves
        #output_cp = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        output_cp = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        cv.drawContours(output_cp, sorted_contours, -1, (0, 255, 0), 30)
        

        #Initialize dataframe
        object_info = []

        for idx, contour in enumerate(sorted_contours):
    
            #Get centroid
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(output_cp, (cx, cy), 40, (255, 0, 255), -1) 
            
            #Get bounding box around each contour
            x_leaf, y_leaf, w_leaf, h_leaf = cv.boundingRect(contour)

            #Draw rectangle and label it
            #cv.rectangle(output_cp, (x, y), (x + w, y + h), (0, 255, 0), 30)
            
            #Show width line
            cv.line(output_cp, (x_leaf, y_leaf + h_leaf//2), (x_leaf + w_leaf, y_leaf + h_leaf//2), (255, 0, 255), 30)  
            
            #Show height line
            cv.line(output_cp, (x_leaf + w_leaf // 2, y_leaf), (x_leaf + w_leaf // 2, y_leaf + h_leaf), (255, 0, 255), 30)
            
            #Get angle from base to middle
            
            #Step 1: Identify base of leaf
            lowest_point = tuple(contour[contour[:, :, 1].argmax()][0])
            cv.circle(output_cp, lowest_point, 45, (255,0,0), -1)  # Blue dot at lowest point
            x0, y0 = lowest_point
            
            #Step 2: Identify widest part of leaf
            #for line 1: from lowest point to left-most point
            leftmost_point = tuple(contour[contour[:,:,0].argmin()][0])
            x1, y1 = leftmost_point
            
            
            #for line 2: from lowest point to right-most point
            rightmost_point = tuple(contour[contour[:,:,0].argmax()][0])
            #cv.circle(output_cp, rightmost_point, 45, (255,0,0), -1)
            x2, y2 = rightmost_point
            
            #Choose highest y point
            #if bool((y + h//2)<y1)
            if bool(y1>y2):
                y1 = y1
            else:
                y1 = y2
            
            
            #draw line 1 
            cv.line(output_cp, (x0,y0), (x1,y1),(255, 0, 0), 30) 
            #draw line 2 
            #cv.line(output_cp, (x0,y0), (x2,y2),(255, 0, 0), 30) 
            #matching leftmost point height
            cv.line(output_cp, (x0,y0), (x2,y1),(255, 0, 0), 30) 
            
            
            #Draw circles
            #leftmost
            cv.circle(output_cp, (x1,y1), 45, (255,0,0), -1)
            #rightmost
            cv.circle(output_cp, (x2,y1), 45, (255,0,0), -1)
            
            
            #Create vectors
            v1 = np.array([x1,y1]) - np.array([x0,y0])
            #matching leftmost point height
            v2 = np.array([x2,y1]) - np.array([x0,y0])

            # Calculate angle in radians (some arithmetic)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            #arithmetic part 2
            angle_rad = np.arccos(cos_theta)

            # Convert to degrees
            angle_deg = np.degrees(angle_rad)
            
            
            # Put id_number at the center
            cv.putText(output_cp, text=str(idx+1), org=(cx, cy), 
                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=7, color=(0,0,255), 
                       thickness=25, lineType=cv.LINE_AA)  # Black text
                
            
            #Add leaf number from label dictionary
            label = file_tags[(file_tags["xmin"]>x_leaf) & 
                  (file_tags["xmax"]<(x_leaf+w_leaf)) &
                  (file_tags["ymin"]>y_leaf) &
                  (file_tags["ymax"]<(y_leaf+h_leaf))
                 ]

            label = label["label"].to_list()
            
            if len(label)==0:
                label = "NA"
            else:
                label = label[0]


            # Store object data
            object_info.append({
                'object_ID': idx + 1,
                'width': w_leaf,        # Width of object
                'height': h_leaf,        # Height of object
                'base_mid_angle': np.rint(angle_deg).astype(int), #Angle between base and middle of leaf
                'leaf_number': label
            })
    


        output_path = os.path.join(output_folder, f'processed_{image_name}.jpg')
        cv.imwrite(output_path, output_cp)
        
        
        df_unit = pd.DataFrame(object_info)
        df_unit.insert(0, 'image', image_name)

        df = pd.concat([df, df_unit], ignore_index=True)
        

        print(f"Done processing {image_name}")


    # Save to CSV in output folder
    df_output_path = os.path.join(output_folder, result)
    df.to_csv(df_output_path, index=False)
        
if __name__ == "__main__":
    main() 
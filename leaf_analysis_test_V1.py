import os, glob
import sys
import numpy as np
import cv2 as cv
import imutils
import pandas as pd

class options:
    def __init__(self):
        # Input image path/filename
        self.image = "."
        # Results path/filename
        self.result = 'leaf_analysis_results.csv'
        # Image output directory path
        self.outdir = "img"
        
def main():
    # Initialize options
    args = options()
    folder,save= input("enter foldername directory and output directory: ").split()
    input_folder=os.path.abspath(folder)
    output_folder=os.path.abspath(save)
    ext="tif"
    files = glob.glob(os.path.join(input_folder, f'*.{ext}'))
    
    df = pd.DataFrame()

    for i in files:
        print(f"Processing: {i}")
        
        #args.image = input_folder+i
        args.image = i
        
        #Read in image
        image = cv.imread(args.image)
        #plt.imshow(image)
        
        #Make grayscale to create binary mask
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)

        #Smoothing to get leaves only
        edged = cv.Canny(gray, 70, 150)
        edged = cv.dilate(gray, None, iterations=1)
        edged = cv.erode(edged, None, iterations=1)
        #plt.imshow(edged)
        
        #Make binary mask to identify objects
        ret, bm = cv.threshold(edged, 150,200, cv.THRESH_BINARY)
        #ret is the threshold used to classify a pixel as black
        #bm is the binary mask made off the image
        #plt.imshow(bm)
        
        #identify all objects in image
        contours = cv.findContours(bm, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        #filter out objects that are not leaves
        min_area = 500000   # Minimum area threshold
        max_area = 10000000   # Maximum area threshold
        filtered_contours = [cnt for cnt in contours if min_area < cv.contourArea(cnt) < max_area]
        
        #Draw leaves
        output = np.zeros_like(bm)
        cv.drawContours(output, filtered_contours, -1, 255, -1)
        #plt.imshow(output)
        
        #sort leaves
        sorted_contours= sorted(filtered_contours, key=cv.contourArea, reverse= True)
        
        #enumerate objects

        #output = cv.cvtColor(bm, cv.COLOR_GRAY2BGR)

        for idx, contour in enumerate(sorted_contours):
            # Draw the contour
            cv.drawContours(output, [contour], -1, (0, 255, 0), 30)

            # Get contour center for label
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Put number at the center
            cv.putText(output, text=str(idx+1), org=(cx, cy), 
                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=7, color=(0, 255, 0), 
                       thickness=25, lineType=cv.LINE_AA)  # Green text

        #plt.imshow(output)
        
        object_info = []  # To store info of each object

        output_cp = cv.cvtColor(output, cv.COLOR_GRAY2BGR)

        cv.drawContours(output_cp, sorted_contours, -1, (0, 255, 0), 30)

        for idx, contour in enumerate(sorted_contours):
            
            #Get centroid
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(output_cp, (cx, cy), 40, (255, 0, 255), -1) 
            
            # Get bounding box around each contour
            x, y, w, h = cv.boundingRect(contour)

            # Optional: Draw rectangle and label it
            cv.rectangle(output_cp, (x, y), (x + w, y + h), (0, 255, 0), 30)
            
            #Show width line
            cv.line(output_cp, (x, y + h//2), (x + w, y + h//2), (255, 0, 255), 30)  
            #cv.putText(output_cp, f'W={w}px', (x + w//2, y + h),
            #           cv.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 255), 10, cv.LINE_AA)
            
            #Show height line
            cv.line(output_cp, (x + w // 2, y), (x + w // 2, y + h), (255, 0, 255), 30)
            #cv.putText(output_cp, f'H={h}px', (x, y + h//2),
            #           cv.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 255), 10, cv.LINE_AA)
            
            #Get angle from base to middle
            
                #Step 1: Identify base of leaf (lowest point in the object
            lowest_point = tuple(contour[contour[:, :, 1].argmax()][0])
            cv.circle(output_cp, lowest_point, 45, (255,0,0), -1)  # Blue dot at lowest point
            x0, y0 = lowest_point
            
                #Step 2: Identify widest part of the leaf
                #for line 1: from lowest point to left-most point
            leftmost_point = tuple(contour[contour[:,:,0].argmin()][0])
            cv.circle(output_cp, leftmost_point, 45, (255,0,0), -1)
            x1, y1 = leftmost_point
            
                #for line 2: from lowest point to right-most point
            rightmost_point = tuple(contour[contour[:,:,0].argmax()][0])
            cv.circle(output_cp, rightmost_point, 45, (255,0,0), -1)
            x2, y2 = rightmost_point
            
            #draw line 1 
            cv.line(output_cp, (x0,y0), (x1,y1),(255, 0, 0), 30) 
            #draw line 2 
            cv.line(output_cp, (x0,y0), (x2,y2),(255, 0, 0), 30) 
            
            
            #Create vectors
            # Create vectors
            v1 = np.array([x1,y1]) - np.array([x0,y0])
            v2 = np.array([x2,y2]) - np.array([x0,y0])

            # Calculate angle in radians (some arithmetic)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            #arithmetic part 2
            angle_rad = np.arccos(cos_theta)

            # Convert to degrees
            angle_deg = np.degrees(angle_rad)
            
            
            # Put id_number at the center
            cv.putText(output_cp, text=str(idx+1), org=(cx, cy), 
                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=7, color=(0,0,0), 
                       thickness=25, lineType=cv.LINE_AA)  # Black text
            
            # Store object data
            object_info.append({
                'Object_ID': idx + 1,
                'width': w,        # Width of object
                'height': h,        # Height of object
                'base_mid_angle': np.rint(angle_deg).astype(int) #Angle between base and middle of leaf
            })
        
        
        processed_image = cv.cvtColor(output_cp, cv.COLOR_BGR2RGB)
        
        filename = os.path.basename(i)
        output_path = os.path.join(output_folder, f'processed_{filename}')
        cv.imwrite(output_path, processed_image)
        
        
        
        df_unit = pd.DataFrame(object_info)
        df_unit.insert(0, 'filename', filename)

        df = pd.concat([df, df_unit], ignore_index=True)
        
        

        print(f"Done processing {i}")



    # Save to CSV in output folder
    df_output_path = os.path.join(output_folder, args.result)
    df.to_csv(df_output_path, index=False)
        
if __name__ == "__main__":
    main() 
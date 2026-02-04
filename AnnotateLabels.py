import os, glob
import cv2 as cv
import matplotlib.pyplot as plt

def main():
    #impath= "/Users/quino070/LeafShapeAnalysis/test"
    #outpath=impath
    impath = "/Users/quino070/LeafShapeAnalysis/handwrittenlabels/images"
    outpath= "/Users/quino070/LeafShapeAnalysis/handwrittenlabels/labels"
    ext="jpg"
    files = glob.glob(os.path.join(impath, f'*.{ext}'))

    for im_path in files:
    	im = cv.imread(im_path)
    	im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    	name = os.path.splitext(os.path.basename(im_path))[0]

    	plt.imshow(im)
    	plt.title(name)
    	plt.axis("off")
    	plt.show(block=False)

    	lab = input("label: ")

    	plt.close()

    	
    	lab_file = f'{outpath}/{name}.txt'

    	with open(lab_file, "w") as file:
    		file.write(lab)


if __name__ == "__main__":
	main()
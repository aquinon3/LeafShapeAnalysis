import os, glob
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

def main():
    #impath= "/Users/quino070/LeafShapeAnalysis/test"
    #outpath=impath
    nalist = pd.read_csv("/Users/quino070/LeafShapeAnalysis/handwrittenlabels/NAlabels.csv")
    impath = "/Users/quino070/LeafShapeAnalysis/handwrittenlabels/images"
    outpath= "/Users/quino070/LeafShapeAnalysis/handwrittenlabels/labels"
    
    #for lab in nalist["lab"].str.lstrip():
    for na in range(len(nalist)):
        im = os.path.join(f'{impath}/{nalist.iloc[na]["lab"].lstrip()}.jpg')
        im = cv.imread(im)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

        plt.imshow(im)
        plt.title(nalist.iloc[na]["file"])
        plt.axis("off")
        plt.show(block=False)

        new_lab = input("new label: ")

        plt.close()

        lab_file = f'{outpath}/{nalist.iloc[na]["lab"].lstrip()}.txt'

        with open(lab_file, "w") as file:
            file.write(new_lab)


if __name__ == "__main__":
	main()
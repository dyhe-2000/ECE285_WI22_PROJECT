import os, cv2
from roipoly import RoiPoly
from roipoly import MultiRoi
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    # read the training images
    folder = 'data/training'
    filename = '0001.jpg'  
  
    categories = ['cone_orange', 'not_cone_orange']
    cone_orange_pixels = []
    not_cone_orange_pixels = []
    
    
    fig, axs = plt.subplots(2, 4, figsize=(9, 3))
    
    count = 0
    for filename in os.listdir(folder): 
      if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder,filename))
        if count < 8:
            axs[count//4,count%4].imshow(img)
        count += 1
        
    plt.show(block=True)
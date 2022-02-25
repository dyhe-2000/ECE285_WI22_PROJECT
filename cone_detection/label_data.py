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

    for filename in os.listdir(folder): 
      if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder,filename))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_0_1 = img.astype(np.float64)/255
        # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv_img_0_1 = hsv_img.astype(np.float64)/255
        
        for i in range(len(categories)):
            print("please draw area for " + categories[i])
            # display the image and use roipoly for labeling
            fig, ax = plt.subplots()
            ax.imshow(img)
            my_roi = RoiPoly(fig=fig, ax=ax, color='r')
              
            # get the image mask
            mask = np.zeros((img.shape[0],img.shape[1])).astype(np.int64)
            try:
                mask = my_roi.get_mask(img)
            except:
                print("no region label")
                
            for j in range(mask.shape[0]):
                for k in range(mask.shape[1]):
                    if mask[j][k]:
                        if i == 0:
                            cone_orange_pixels.append(img_0_1[j][k].reshape(3,1))
                        if i == 1:
                            not_cone_orange_pixels.append(img_0_1[j][k].reshape(3,1))
    
    cone_orange_training_data = np.zeros((len(cone_orange_pixels),3))
    not_cone_orange_training_data = np.zeros((len(not_cone_orange_pixels),3))
    
    for i in range(len(cone_orange_pixels)):
        cone_orange_training_data[i] = cone_orange_pixels[i].reshape(1,3)
    for i in range(len(not_cone_orange_pixels)):
        not_cone_orange_training_data[i] = not_cone_orange_pixels[i].reshape(1,3)
        
    print(cone_orange_training_data.shape)
    print(not_cone_orange_training_data.shape)
        
    np.save('cone_orange_training_data.npy', cone_orange_training_data)
    np.save('not_cone_orange_training_data.npy', not_cone_orange_training_data)
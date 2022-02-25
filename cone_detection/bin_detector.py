import numpy as np
import cv2
from skimage.measure import label, regionprops
import math
import os
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops, find_contours
import sys
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)

def multi_dil(im, num, element):
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im, num, element):
    for i in range(num):
        im = erosion(im, element)
    return im

class BinDetector():
    def __init__(self):
        '''
            Initilize your bin detector with the attributes you need,
            e.g., parameters of your classifier
        '''
        try:
            # logistic regression
            self.categories = ['cone_orange', 'not_cone_orange']
            folder_path = os.path.dirname(os.path.abspath(__file__)) 
            self.omega_class_list = []
            
            for i in range(len(self.categories)):
                omega_class_params_file = os.path.join(folder_path, 'omega_' + self.categories[i] + '.npy')
                self.omega_class_list.append(np.load(omega_class_params_file))
        except:
            pass

    def segment_image(self, img):
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        # Replace this with your own approach
        original_img = img
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img_0_1 = rgb_img.astype(np.float64)/255
        img_0_1 = img.astype(np.float64)/255
        mask_img = np.zeros((img.shape[0],img.shape[1])).astype(bool)
        
        
        for i in range(img.shape[0]): # iterating through samples
            for j in range(img.shape[1]):
                current_pixel_rgb = img_0_1[i][j].reshape(1,3)
                        
                # rgb classification
                
                picking_max_arg_array = np.zeros((len(self.categories),1))
                
                for k in range(len(self.omega_class_list)):
                    picking_max_arg_array[k][0] = np.matmul(current_pixel_rgb, self.omega_class_list[k])[0][0]
                
                max_rgb_index = 0
                max_rgb_val = picking_max_arg_array[0][0]
                for k in range(picking_max_arg_array.shape[0]):
                    if picking_max_arg_array[k][0] > max_rgb_val:
                        max_rgb_index = k
                        max_rgb_val = picking_max_arg_array[k][0]
                
                if max_rgb_index == 0 and max_rgb_val > 745:
                    mask_img[i][j] = 1
                        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Horizontally stacked subplots')
        ax1.imshow(rgb_img_0_1)
        ax2.imshow(mask_img)
        plt.show()
        
        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return mask_img

    def get_bounding_boxes(self, img):
        '''
            Find the bounding boxes of the recycling bins
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        
        # Replace this with your own approach 
        element = np.array([[0,0,0,0,0,0,0],
                    [0,0,1,1,1,0,0],
                    [0,1,1,1,1,1,0],
                    [0,1,1,1,1,1,0],
                    [0,1,1,1,1,1,0],
                    [0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,0]])
                    
        boxes = []
        
        img = multi_ero(img, 2, element)
        img = multi_dil(img, 2, element)
        
        
        mask_labels = label(img)
        props = regionprops(mask_labels)

        img_copy = np.copy(img)*255
        for prop in props:
            if prop.bbox_area > 0:
                if (prop.bbox[2]-prop.bbox[0])/(prop.bbox[3]-prop.bbox[1]) > 1.1 and (prop.bbox[2]-prop.bbox[0])/(prop.bbox[3]-prop.bbox[1]) < 1.8:
                    cv2.rectangle(img_copy, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255,255,255), 5)
                    boxes.append([prop.bbox[1],prop.bbox[0],prop.bbox[3],prop.bbox[2]])
        
        # YOUR CODE BEFORE THIS LINE
        ################################################################
        
        return boxes
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def train(self, dataset):
        '''
            The training code. Include how you process data, how to train the model.
            Can be multiple functions.
        '''
        X = dataset[0]
        y = dataset[1]
        
        self.categories = ['cone_orange', 'non_cone_orange'] # 1,2
        
        y_class_vs_all_list = [] # contains y_class_i_vs_all
        for i in range(len(self.categories)):
            y_class_i_vs_all = np.zeros((y.shape[0],))
            for j in range(y.shape[0]):
                if y[j] == i+1:
                    y_class_i_vs_all[j] = 1
                else:
                    y_class_i_vs_all[j] = -1
            y_class_vs_all_list.append(np.copy(y_class_i_vs_all))
        
        omega_class_list = []
        for i in range(len(self.categories)):
            omega_i = np.zeros((3,1))
            omega_class_list.append(np.copy(omega_i))
        
        learning_rate = 0.1
        
        print("data len: " + str(X.shape[0]))
        
        for t in range(1000):
            print(t)
            
            for j in range(len(self.categories)):
                summation = np.zeros((3,1));
                for i in range(X.shape[0]):
                    summation = summation + (1-self.sigmoid(y_class_vs_all_list[j][i]*np.matmul(X[i].reshape(1,3),omega_class_list[j])[0][0]))*y_class_vs_all_list[j][i]*(X[i].reshape(3,1))
                omega_class_list[j] = omega_class_list[j] + learning_rate*summation;
            
        self.omega_class_list = omega_class_list
        
        for i in range(len(self.omega_class_list)):
            print('omega_' + self.categories[i] + '.npy')
            print(self.omega_class_list[i])
            np.save('omega_' + self.categories[i] + '.npy', self.omega_class_list[i])

if __name__ == '__main__':
    cone_orange_training_data = np.load('cone_orange_training_data.npy')
    not_cone_orange_training_data = np.load('not_cone_orange_training_data.npy')
    
    cone_orange_training_data_reduced = cone_orange_training_data[::1]
    not_cone_orange_training_data_reduced = not_cone_orange_training_data[::1]
    
    print(cone_orange_training_data_reduced.shape)
    print(not_cone_orange_training_data_reduced.shape)
    
    y1 = np.full(cone_orange_training_data_reduced.shape[0],1)
    y2 = np.full(not_cone_orange_training_data_reduced.shape[0],2)
    X = np.concatenate((cone_orange_training_data_reduced,not_cone_orange_training_data_reduced))
    y = np.concatenate((y1,y2))
    

    bin_detector = BinDetector()
    # bin_detector.train([X, y])
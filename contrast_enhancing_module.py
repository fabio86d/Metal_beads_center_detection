# DIFFERENT RESCALING METHODS TO ENHANCE CONTRAST

import cv2
import numpy as np
import matplotlib.pyplot as plt


# CONTRAST STRETCHING (SLOWER METHOD)
# ATT: it returns somehow a (width, height, 1) image. You have to squeeze() it to display it with plt.imshow()
def rescale_img(img, min, max, output_depth = 256):

    print(''' Rescale image function >>> ATT: a (width, height, 1) image is returned. 
    You have to img.squeeze() it to display it with plt.imshow() ''')

    rows, cols = img.shape
    step = (max - min)/output_depth

    palette = np.zeros((2**16,1), np.uint8)
    palette[max:,:] = output_depth - 1
    for i in range(output_depth):
        palette[(min+i*step):(min+(i+1)*step)] = i

    output = palette[img]                                
     
    return output



# LOOK UP TABLE: FASTER CONTRAST STRETCHING METHOD (from 16bit to 8bit ONLY)
def clip_and_rescale(img, min, max):

    image = np.array(img, copy = True) # just create a copy of the array
    image.clip(min,max, out = image)
    image -= min
    #image //= (max - min + 1)/256.
    image = np.divide(image,(max - min + 1)/256.)
    return image.astype(np.uint8)

def look_up_table(image, min, max):

    lut = np.arange(2**16, dtype = 'uint16')  # lut = look up table
    lut = clip_and_rescale(lut, min, max)

    return np.take(lut, image)  # it s equivalent to lut[image] that is "fancy indexing"




# execution
if __name__ == "__main__":

    file_dir = r'D:\Geneva_phantom\ready_images\3mm_side\18cm_distance\p22\18d_3mm_90v_22_001.tif'
    img16 = cv2.imread(file_dir, cv2.IMREAD_ANYDEPTH) # 16 bit image
    img8 = cv2.imread(file_dir, 0) # 8 bit image 

    # executions
    cv2.imshow('original image', img16)
    #cv2.imshow('rescaled image', rescale_img(img16, 8992, 54816 ))
    cv2.imshow('rescaled image look up table', look_up_table(img16, 8992, 54816))
    cv2.imshow('equalized', cv2.equalizeHist(img8))
    clahe = cv2.createCLAHE()
    cv2.imshow('adaptive equalization', clahe.apply(img8))
    cv2.waitKey(0)

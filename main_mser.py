import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from contrast_enhancing_module import look_up_table
from ROI_module import ROIbuilder_square
from read_and_write import write_nparray_to_xlsx

# define callback function
def detect_mser(x):
    
    global img, img_contrast, img_mser, img_mser_zoomed, regions

    
    # get current canny parameters
    delta = cv2.getTrackbarPos('delta','Control Panel')
    minArea = cv2.getTrackbarPos('minArea','Control Panel')
    maxArea = cv2.getTrackbarPos('maxArea','Control Panel')
    maxVariation = cv2.getTrackbarPos('maxVariation','Control Panel')
    minDiversity = cv2.getTrackbarPos('minDiversity','Control Panel')
    #print 'delta', delta, 'minArea', minArea, 'maxArea', maxArea, 'minDiversity', minDiversity
    min_contrast = cv2.getTrackbarPos('minContrast','Control Panel')
    max_contrast = cv2.getTrackbarPos('maxContrast','Control Panel')

    img_contrast = look_up_table(img, min_contrast, max_contrast)
    vis = cv2.cvtColor(img_contrast, cv2.COLOR_GRAY2BGR)

    mser = cv2.MSER_create(_delta = delta, _min_area = minArea, _max_area = maxArea, _max_variation = np.divide(float(maxVariation),1000.0), _min_diversity = np.divide(float(minDiversity),100.0))
    #regions = mser.detectRegions(img_contrast,None)
    regions, _ = mser.detectRegions(img_contrast) # for new version of opencv2

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions] # regions is a list of (numpy) arrays. each of those arrays is variableNX2 sized. you transform it into a NX1X2 (list?)
    img_mser = cv2.polylines(vis, hulls, 1, (0, 255, 0))

    img_mser_zoomed = cv2.resize(img_mser[pp[0][1]: pp[1][1] ,pp[0][0]: pp[1][0]], (int(original_height*factor), int(original_width*factor)), interpolation=cv2.INTER_LINEAR )


############################

# load image as 16 bit
file_dir = sys.argv[1]
img = cv2.imread( file_dir, cv2.IMREAD_ANYDEPTH) # 16 bit image cv2.IMREAD_ANYDEPTH
min_img_contrast = np.amin(img)
max_img_contrast = np.amax(img)
#min_contrast = 8992
#max_contrast = 54816
#min_contrast = 8000
#max_contrast = 40000
# rescale 16bit into 8bit with contrast stretching
img_for_display = look_up_table(img, min_img_contrast, max_img_contrast)
img_for_contrast = np.copy(img_for_display)
display_image = cv2.cvtColor(img_for_display, cv2.COLOR_GRAY2BGR)
img_mser = img
img_mser_zoomed = img
regions = []

window_name_ROI = "Select ROI in direction top-left to bottom-right. Press Esc button to finish"
window_name_zoomed_img = "Zoomed Image"
window_name_img = "Entire Image"

original_height, original_width = img.shape[:2]
factor = 0.5

cv2.namedWindow(window_name_ROI)
cv2.imshow(window_name_ROI, img_for_contrast)

# detect ROIs
rois = ROIbuilder_square(window_name_ROI, img_for_contrast)
num_metal_beads = len(rois.refPt_set)    
 


######################
## initialize image
#img_mser_zoomed = img

## create window
#cv2.namedWindow('image')
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)

## create trackbars for color change (http://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x)
cv2.createTrackbar('delta','Control Panel',5,50,detect_mser)  # Delta delta, in the code, it compares (size_{i}-size_{i-delta})/size_{i-delta}. default 5
cv2.createTrackbar('minArea','Control Panel',60,200,detect_mser)  # MinArea prune the area which smaller than minArea. default 60.
cv2.createTrackbar('maxArea','Control Panel',1300,1500,detect_mser)  # MaxArea prune the area which bigger than maxArea. default 14400.
cv2.createTrackbar('maxVariation','Control Panel',250,1000,detect_mser) # f a region is maximally stable, it can still be rejected if the the regions variation is bigger than maxVariation. default .25
cv2.createTrackbar('minDiversity','Control Panel',20,1000,detect_mser) # trace back to cut off mser with diversity < min_diversity. default 0.2.
cv2.createTrackbar('minContrast','Control Panel',0,65535,detect_mser) # min constrast img
cv2.createTrackbar('maxContrast','Control Panel',65535,65535,detect_mser) # min constrast img

# generate array for outputs
centers = np.zeros((num_metal_beads,2), dtype = float)


for i in range(num_metal_beads):

    flag_single_centre = True
    while flag_single_centre:

        pp = rois.refPt_set[i]
        m = rois.masks_set[i]
        img_mser_zoomed = cv2.resize(img[pp[0][1]: pp[1][1] ,pp[0][0]: pp[1][0]], (int(original_height*factor), int(original_width*factor)), interpolation=cv2.INTER_LINEAR )
        img_contrast = look_up_table(img, 0, 65535)

        while(1):

            cv2.imshow(window_name_zoomed_img,img_mser_zoomed)
            cv2.imshow(window_name_img,img_contrast)
            #k = cv2.waitKey(0)
            #if k == ord('r'):
            #    break

            k = cv2.waitKey(1) & 0xFF
            if k == 27:

                mm = []
                for j in regions:
                    mm.append(np.array([j[:,1],j[:,0]]))

                # select only the regions included in the current mask (of the current selected bead)
                real_regions = np.asarray([regions[k] for k in range(len(regions)) if np.all(m[tuple(mm[k])] == 1)])
                #print 'length original regions', len(regions), 'length new regions', len(real_regions)
                   
                if len(real_regions) == 1 :
                    ellipse = cv2.fitEllipse(real_regions)
                    # centers are saved at sub pixel accuracy, but they are displayed in the image at pixel accuracy
                    centers[i] = ellipse[0]
                    #print np.around(centers[i]).astype(int)
                    print(centers[i])
                    #cv2.ellipse(display_image,ellipse,(0,255,0),1)
                    cv2.circle(display_image, tuple(centers[i].astype(int)), 1, (0,0, 65520))
                    #cv2.circle(display_image, tuple(centers[i]), 1, (0,0, 255))
                    flag_single_centre = False

                if len(real_regions) > 1 :

                        print("More than one contour has been selected for one metal bead")
                
                if len(real_regions) == 0 :

                        print("No centre was detected")
                        #centers[i] = (float("inf"), float("inf"))
                        #centers[i] = (np.nan, np.nan)
                        centers[i] = (0.0, 0.0)
                        flag_single_centre = False
                break

        # just to show where (0,0) is
        cv2.circle(display_image, tuple((0,0)), 1, (0,0, 65520))

        cv2.destroyWindow(window_name_zoomed_img)
        cv2.destroyWindow(window_name_img)

# draw centers
cv2.imshow('fit ellipse real regions', display_image)
cv2.waitKey(0)


# write outputs
save_dir = file_dir.split('.')[0]
save_name = save_dir.split('\\')[-1]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
write_nparray_to_xlsx(centers, save_dir + '/' + save_name + '_detected_centers' + '.xlsx', 'metal_beads_centers')

# write image
cv2.imwrite(save_dir + '/' + save_name + 'figure_centers.tif', display_image)

plt.show()

#for i in len(rois.refPt_set)

#while(1):

#    cv2.imshow('image',img_mser)
#    k = cv2.waitKey(1) & 0xFF
#    if k == 27:
#        break


cv2.destroyAllWindows()


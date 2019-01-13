import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import numpy.ma as ma
from operator import add

class ROIbuilder:

    def __init__(self,ax, nr, nc):

        self.ax = ax
        self.nr = nr
        self.nc = nc

        # callbacks
        self.idpress_init = self.ax.figure.canvas.mpl_connect('button_press_event', self.starting_point)
        self.idpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.button_press_callback)


    def starting_point(self,event):
        print("starting point")
        xs_init = event.xdata
        ys_init = event.ydata
        line, = self.ax.plot([xs_init], [ys_init])
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.mask = []
        self.ax.figure.canvas.mpl_disconnect(self.idpress_init)

    def get_binary_mask(self):
        print("calculating mask")
        xycrop = np.vstack((self.xs, self.ys)).T
        pth = Path(xycrop, closed=False)
        ygrid, xgrid = np.mgrid[:self.nr, :self.nc]
        xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
        mask = pth.contains_points(xypix)
        mask = mask.reshape((self.nr,self.nc))
        self.mask = mask

    def button_press_callback(self, event):
        
        if self.line == None:
            return

        if event.inaxes!=self.line.axes: return

        if event.button != 1: 
            self.line.figure.canvas.mpl_disconnect(self.idpress)
            self.get_binary_mask()
            print('disconnected. Line is', self.line)
            return

        # print 'click', event
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        # print 'the line', self.line, 'contains', self.xs, self.ys



class ROIbuilder_rect:

    def __init__(self,window_name, image):

        print("init is declared")
        self.window_name = window_name
        self.image = image.copy()
        self.refPt = [None]*2
        self.refPt_set = []
        self.mask = np.zeros(image.shape, np.bool)
        self.masks_set = []
        #self.rois_set = []

        cv2.setMouseCallback(window_name, self.draw_rect_roi)
        
        while True:
            key = cv2.waitKey(0)

            if key == ord("c"):

                cv2.destroyAllWindows()

                # display masks set
                #for i in self.masks_set:

                #    cv2.imshow("mask", i.astype(np.float))
                #    cv2.waitKey(0)

                #cv2.destroyAllWindows()

                break

    def draw_rect_roi(self, event, x, y, flags, param):
 
        #print "draw_rect_roi is called"
        #print 'refPt_set', self.refPt_set
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.refPt[0] = (x, y)
 
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt[1] = (x,y)

            self.refPt_set.append(self.refPt)

            # update mask
            self.mask[self.refPt[0][1]: self.refPt[1][1] , self.refPt[0][0]: self.refPt[1][0]] = True

            # update masks_set
            self.masks_set.append(self.mask)

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)

            self.refPt = [None]*2
            self.mask = np.zeros(self.image.shape, np.bool)
        


class ROIbuilder_square:

    def __init__(self,window_name, image):

        print("init is declared")
        self.window_name = window_name
        self.image = image.copy()
        self.refPt = [None]*2
        self.refPt_set = []
        self.mask = np.zeros(image.shape, np.bool)
        self.masks_set = []
        #self.rois_set = []

        cv2.setMouseCallback(window_name, self.draw_rect_roi)
        
        while True:
            key = cv2.waitKey(0)

            if key == 27:

                cv2.destroyAllWindows()

                # display masks set
                #for i in self.masks_set:

                #    cv2.imshow("mask", i.astype(np.float))
                #    cv2.waitKey(0)

                #cv2.destroyAllWindows()

                break

    def draw_rect_roi(self, event, x, y, flags, param):
 
        #print "draw_rect_roi is called"
        #print 'refPt_set', self.refPt_set
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.refPt[0] = (x, y)
 
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            side = np.min((x - self.refPt[0][0],y - self.refPt[0][1]))
            self.refPt[1] = (self.refPt[0][0] + side ,self.refPt[0][1] + side)

            self.refPt_set.append(self.refPt)

            # update mask
            self.mask[self.refPt[0][1]: self.refPt[1][1] , self.refPt[0][0]: self.refPt[1][0]] = True

            # update masks_set
            self.masks_set.append(self.mask)

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)

            self.refPt = [None]*2
            self.mask = np.zeros(self.image.shape, np.bool)




if __name__ == "__main__":
    
    ## load image
    #img = cv2.imread(sys.argv[1], 0) # 8 bit image 

    #fig = plt.figure()
    #ax = fig.add_subplot(111) # a way to get the axes
    #ax.set_title('click to build ROI')
    #plt.imshow(img,cmap = 'gray',interpolation = 'none')
    #plt.xticks([]), plt.yticks([])
    #roi = ROIbuilder(ax,img.shape)
    #plt.show()

    #plt.imshow(roi.mask,cmap = 'gray',interpolation = 'none')
    #plt.show()

    #print 'Vertices: x coordinates', roi.xs
    #print 'Vertices: y coordinates', roi.ys



    file_dir = r'F:\Geneva_phantom\ready_images\3mm_side\18cm_distance\p22\18d_3mm_90v_22_001.tif'
    image = cv2.imread(file_dir, cv2.IMREAD_ANYDEPTH) # 16 bit image cv2.IMREAD_ANYDEPTH

    cv2.namedWindow("image")
    cv2.imshow("image", image)
    print(image.shape)
    #roi = ROIbuilder_rect("image",image)
    roi = ROIbuilder_square("image",image)

    #while True:
    #    key = cv2.waitKey(0)

    #    if key == ord("c"):

    #        cv2.destroyAllWindows()
    #        break

    #for i in roi.masks_set:

    #    cv2.imshow("mask", i.astype(np.float))
    #    cv2.waitKey(0)
    #print 'masks' , len(roi.masks_set)

    #cv2.namedWindow("mask")
    #cv2.imshow("mask", roi.mask.astype(np.float))
    
    cv2.destroyAllWindows()
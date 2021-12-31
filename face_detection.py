from commonfunctions import *
import numpy as np
import cv2
import math
import timeit
from multiprocessing import Process
import joblib

filename = 'finalized_model.sav'
finalized_model_1 = joblib.load(filename)

def cut_image(img,windowsize_r,windowsize_c):
    tiles=[]
    img_r=img.shape[0]
    img_c=img.shape[1]
    tiles=img.reshape(img_r//windowsize_r,
                      windowsize_r,
                     img_c//windowsize_c,
                     windowsize_c)
    tiles=tiles.swapaxes(1,2)
    return tiles

#cell is a matrix 8x8
def gradient(img):
    img=np.float32(img) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

#     show_images([gx,gy,mag,angle],["gx","gy","mag","angle"])
#     print("mag",np.sum(mag))
    return mag,angle%180


def HOG_histogram(mag, angle, bin_size=20):
    start = timeit.default_timer()
    # mag=mag_angle[:mag_angle.shape[0]//2]
    # angle=mag_angle[mag_angle.shape[0]//2:]
    hist = np.zeros(int(180 / bin_size))
    low_bins = 20 * (angle // 20)

    high_bins = 20 + low_bins

    ## weights
    weight_low_bin = (abs(high_bins - angle) / 20) * mag
    weight_high_bin = (abs(angle - low_bins) / 20) * mag

    high_bins = high_bins % 180
    for i in range(hist.shape[0]):
        votes_low = np.sum(weight_low_bin[(low_bins // 20) == i])
        votes_high = np.sum(weight_high_bin[(high_bins // 20) == i])
        hist[i] = votes_low + votes_high
    stop = timeit.default_timer()
    # print(stop-start)
    return hist


#     print(np.sum(mag))

#     print(np.sum(hist))

def HOG_descriptor(img):
    resized = cv2.resize(img, (48, 64), interpolation=cv2.INTER_AREA)
    mag, angle = gradient(resized)
    #     print(resized.shape)

    # loop on mag and angle to get 16*16 cell
    windowsize_r = 16
    windowsize_c = 16
    bin_size = 20
    mag_cells = []
    ang_cells = []

    # check code image on what`s app for efficient split

    # TODO make 2d

    #     for r in range(0,mag.shape[0] - windowsize_r, windowsize_r):
    #         temp_mag=[]
    #         temp_angle=[]
    #         for c in range(0,mag.shape[1] - windowsize_c, windowsize_c):

    #             temp_mag.append(mag[r:r+windowsize_r,c:c+windowsize_c])
    #             temp_angle.append(angle[r:r+windowsize_r,c:c+windowsize_c])

    #         mag_cells.append(temp_mag)
    #         ang_cells.append(temp_angle)

    # get 16*16 cells

    start = timeit.default_timer()
    mag_cells = cut_image(mag, windowsize_r, windowsize_c)
    ang_cells = cut_image(angle, windowsize_r, windowsize_c)
    # print((mag_cells.shape[0],mag_cells.shape[1],-1))
    # mag_cells=np.resize(mag_cells,(mag_cells.shape[0],mag_cells.shape[1],-1))
    # ang_cells=np.resize(ang_cells,(mag_cells.shape[0],mag_cells.shape[1],-1))
    # mag_ang_cells=np.concatenate([mag_cells, ang_cells],axis=-1)
    # print(mag_ang_cells.shape)
    # stop = timeit.default_timer()
    # print("cutting",stop-start)

    #     print(mag_cells.shape)
    #     show_images([mag_cells[1][0],mag_cells[1][1],mag_cells[1][2],mag_cells[1][3]])

    # histogram of each cell
    hist = np.zeros((mag_cells.shape[0], mag_cells.shape[1], 180 // bin_size))
    # start = timeit.default_timer()
    # print(mag_cells.shape)
    start = timeit.default_timer()
    for r in range(mag_cells.shape[0]):
        for c in range(mag_cells.shape[1]):
            hist[r][c] = HOG_histogram(mag_cells[r][c], ang_cells[r][c], bin_size)
    stop = timeit.default_timer()
    # print("hist",stop-start)
    # start = timeit.default_timer()
    # hist=np.apply_along_axis(HOG_histogram, -1, mag_ang_cells)
    # stop = timeit.default_timer()
    # print(stop-start)

    start = timeit.default_timer()
    norm_hist = []
    for r in range(hist.shape[0] - 1):
        for c in range(hist.shape[1] - 1):
            temp = hist[r:r + 2, c:c + 2].reshape(-1)
            temp /= np.linalg.norm(temp)
            norm_hist += list(temp)
    norm_hist = np.asarray(norm_hist)
    stop = timeit.default_timer()
    # print("averaging",stop-start)
    #     print('size of hist',hist.shape)
    #     print('size of norm_hist',norm_hist.shape)
    #     print(hist)
    return norm_hist


def search_for_face(img,division_factor1,division_factor2,step):
    partitions = []
    x,y=img.shape
    l,w=(int(img.shape[0]/division_factor1),int(img.shape[1]/division_factor2))
    print("mememem")
    for row in range(0,x-l,step):
        for col in range(0,y-w,step):
            #print(l,w)
            #print("hehhh")
            partition=img[row:row+l,col:col+w]
            prediction=finalized_model_1.predict(HOG_descriptor(cv2.resize(partition, (48,64), interpolation = cv2.INTER_AREA)).reshape(1, -1),)
            if prediction!=0:
                print(prediction)
                show_images([partition])
                partitions.append(partition)
    return partitions
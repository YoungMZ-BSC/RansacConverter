import csv
import cv2
from matplotlib import pyplot as plt
from numpy import unique
from numpy import where
from sklearn.mixture import GaussianMixture
from sklearn import linear_model, datasets
from skimage.measure import LineModelND, ransac

import numpy as np
import math
import os

def morphological_gradient(img,iterr = 1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    erosion = cv2.erode(img, kernel, iterations = iterr)
    dilation = cv2.dilate(img, kernel, iterations= iterr)
    edge = dilation - erosion
    #edge is the final mask with the morphological operation

    return edge
def get_topbottom(edge):
    listofwhite = np.where(edge[:,:] != 0)

    leftmostindex = np.where(listofwhite[1] == np.amin(listofwhite[1]))
    leftpoints = np.array([[]])
    for i in leftmostindex[0]:
        if leftpoints.shape == (1,0):
            leftpoints = np.concatenate((leftpoints,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 1)
        else:
            leftpoints = np.concatenate((leftpoints,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 0)

    if leftpoints.shape != (1,2):
        leftpoint = leftpoints[int(len(leftpoints)/2)][:]

    rightmostpixel = np.where(listofwhite[1] == np.amax(listofwhite[1]))
    rightpoints = np.array([[]])
    for i in rightmostpixel[0]:
        if rightpoints.shape == (1,0):
            rightpoints = np.concatenate((rightpoints,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 1)
        else:
            rightpoints = np.concatenate((rightpoints,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 0)

    model = GaussianMixture(n_components=2)
    # fit the model
    model.fit(rightpoints)
    # assign a cluster to each example
    yhat = model.predict(rightpoints)
    clusters = unique(yhat)

    singlepoints_R = []
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        rightpoints_temp = rightpoints[row_ix[0]]
        singlepoints_R += [rightpoints_temp[int(len(rightpoints_temp)/2)]]
    middlepoint_R = [(singlepoints_R[0][0]+singlepoints_R[1][0])/2,singlepoints_R[0][1]]

    #defining the linear line between it
    #m is slope, b is y intercept
    if middlepoint_R[0]-leftpoint[0] == 0:
        m = 10000000
    else:
        m = (middlepoint_R[1]-leftpoint[1])/(middlepoint_R[0]-leftpoint[0])
    b = int(middlepoint_R[1]-m*middlepoint_R[0])

    listofwhite_top=np.array([[]])
    listofwhite_bottom=np.array([[]])

    for i in range(len(listofwhite[0])):
        pair = listofwhite[0][i],listofwhite[1][i]
        y = m*listofwhite[0][i] + b
        if listofwhite[1][i]-y>=0:
            if listofwhite_top.shape == (1,0):
                listofwhite_top = np.concatenate((listofwhite_top,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 1)
            else:
                listofwhite_top = np.concatenate((listofwhite_top,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 0)
        else:
            if listofwhite_bottom.shape == (1,0):
                listofwhite_bottom = np.concatenate((listofwhite_bottom,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 1)
            else:
                listofwhite_bottom = np.concatenate((listofwhite_bottom,[[listofwhite[0][i],listofwhite[1][i]]]),axis = 0)


    return listofwhite_top,listofwhite_bottom


def convertlisttoImage(top,bottom,img_shape):
    img = np.zeros((img_shape[0],img_shape[1],3))
    for i in top:
        img[int(i[0]),int(i[1]),0] = 255
    for i in bottom:
        img[int(i[0]),int(i[1]),1] = 255
    return img

def custom_loss(y_true,y_pred, loss="absolute_loss"):

    if y_true.ndim == 1:
        if loss == "absolute_loss":
            loss_function = np.abs(y_true - y_pred)
        elif loss =="squared_loss":
            loss_function =(y_true - y_pred) ** 2

    else:
        y_min = np.min(y_true)
        y_max = np.max(y_true)
        weight = np.zeros((np.shape(y_true)))
        b = 2
        for i in range(len(y_true)):
            num = 10 - (np.exp(b)*10)
            den = (np.exp(b)*y_max) - y_min
            a = num/den
            ax = a * y_true[i]
            logx = ax +10
            c = -np.log10(a*y_max+10)
            # m = (-b)/(y_max-y_min)
            # c = b-(m*y_min)
            # weight[i] = m *y_true[i] + c
            weight[i] = np.log10(logx) + c
        if loss == "absolute_loss":
            loss_function = np.sum(np.abs((y_true - y_pred)*weight), axis=1)
        elif loss == "squared_loss":
            loss_function = np.sum((((y_true - y_pred) ** 2)*weight), axis=1)

    return loss_function

def RANSAC(edge,img1_shape):
    X = edge.T[0].reshape(-1,1)
    y = edge.T[1].reshape(-1,1)
    # Robustly fit linear model with RANSAC algorithm
    # loss_function = custom_loss(y_true, y_pred, loss="absolute_loss")
    # ransac = linear_model.RANSACRegressor(min_samples=20, max_trials=1000, loss=custom_loss)
    # ransacc = linear_model.RANSACRegressor(min_samples=10, max_trials=1000, residual_threshold=5, loss=custom_loss)

    data = np.column_stack([X, y])

    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                                   residual_threshold=1, max_trials=1000)
    outliers = inliers == False
    #
    #
    # ransacc.fit(X, y)
    # inlier_mask = ransacc.inlier_mask_
    # outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(0, 400)
    line_y_ransac = model_robust.predict(line_X).T[1].reshape(-1,1)


    line_Xx = np.arange(0, img1_shape[0])[:, np.newaxis]
    # line_yy_ransac = ransacc.predict(line_Xx)


    return X,y,inliers,outliers,line_Xx, line_y_ransac

def findinliers(X,y,line_X, line_y_ransac):
    XY_real= np.concatenate((X, y), axis=1)
    XY_real_ordered = [tuple(i) for i in XY_real]
    dtype = [('xaxis', int), ('yaxis', int)]
    XY_real_ordered = np.array(XY_real_ordered, dtype=dtype)
    XY_real_ordered = np.sort(XY_real_ordered, order='yaxis')
    XY_real_ordered = np.array([list(i) for i in XY_real_ordered])


    XY_inliers = []
    XY_outliers = []
    for i in reversed(XY_real_ordered):
        p3 = np.array((i[0],i[1]))
        p1 = np.array((line_X[0][0],line_y_ransac[0][0]))
        p2 = np.array((line_X[-1][0],line_y_ransac[-1][0]))
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

        if d <= 2:
            XY_inliers.append(i)
        else:
            XY_outliers.append(i)
    return XY_inliers,XY_outliers, XY_real_ordered


def draw(img, XY_inliers, XY_outliers, line_X, line_y_ransac, subplot):
    lw = 1

    for i in XY_inliers:
        img[int(i[0]), int(i[1]), 0] = 255

    for i in XY_outliers:
        img[int(i[0]), int(i[1]), 1] = 255

    subplot.imshow(img)
    subplot.plot(
        line_y_ransac,
        line_X,
        color="cornflowerblue",
        linewidth=lw,
        label="RANSAC regressor",
    )

    inliers_minx = np.amin(XY_inliers, axis=0)[0]
    inliers_miny = np.amin(XY_inliers, axis=0)[1]
    inliers_maxx = np.amax(XY_inliers, axis=0)[0]
    inliers_maxy = np.amax(XY_inliers, axis=0)[1]

    outliers_minx = np.amin(XY_outliers, axis=0)[0]
    outliers_miny = np.amin(XY_outliers, axis=0)[1]
    outliers_maxx = np.amax(XY_outliers, axis=0)[0]
    outliers_maxy = np.amax(XY_outliers, axis=0)[1]

    finalmin_x = min(inliers_minx, outliers_minx)
    finalmin_y = min(inliers_miny, outliers_miny)
    finalmax_x = max(inliers_maxx, outliers_maxx)
    finalmax_y = max(inliers_maxy, outliers_maxy)

    return finalmin_x, finalmin_y, finalmax_x, finalmax_y


def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def filter(XY_inliers,XY_outliers):  # to remove inliers far away that lie on the RANSAC line.

    XY_inliers_final = []
    XY_outliers_final = []
    midpoint = []

    for i in range(len(XY_inliers) - 1, 1, -1):
        pp1 = [XY_inliers[i - 1][0], XY_inliers[i - 1][1]]
        pp2 = [XY_inliers[i][0], XY_inliers[i][1]]
        dist = math.dist(pp1, pp2)
        if dist >= 10:
            midpoint = XY_inliers[i-5]

    if midpoint != []:
        for i in XY_inliers:
            if i[1] <= midpoint[1]:
                XY_outliers.append(list(i))
            else:
                XY_inliers_final.append(list(i))

    else:
        XY_inliers_final = XY_inliers

    return XY_inliers_final, XY_outliers

def entirePipeline(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_shape = np.shape(img)
    figure = plt.figure(figsize=(60, 20), dpi=30)
    subplot1 = figure.add_subplot(1, 5, 1)

    subplot1.imshow(img, cmap='gray')
    subplot1.title.set_text("Gray Scale")

    iterr = 1
    edge = morphological_gradient(img, iterr)
    subplot2 = figure.add_subplot(1, 5, 2)
    subplot2.imshow(edge, cmap='gray')
    subplot2.title.set_text("Edge")

    Top, Bottom = get_topbottom(edge)
    img = convertlisttoImage(Top, Bottom, img_shape)
    subplot3 = figure.add_subplot(1, 5, 3)
    subplot3.imshow(img, cmap='gray')
    subplot3.title.set_text("Top and bottom separated")

    X1, y1, inlier_mask, outlier_mask, line_X1, line_y_ransac1= RANSAC(Top, img_shape)
    XY_inliers1, XY_outliers1, XY_real_ordered1 = findinliers(X1, y1, line_X1, line_y_ransac1)
    X2, y2, inlier_mask, outlier_mask, line_X2, line_y_ransac2 = RANSAC(Bottom, img_shape)
    XY_inliers2, XY_outliers2, XY_real_ordered2 = findinliers(X2, y2, line_X2, line_y_ransac2)
    subplot4 = figure.add_subplot(1, 5, 4)
    img = np.zeros((img_shape[0], img_shape[1], 3))
    finalmin_x, finalmin_y, finalmax_x, finalmax_y = draw(img, XY_inliers1, XY_outliers1, line_X1, line_y_ransac1,
                                                          subplot4)
    finalmin_x1, finalmin_y1, finalmax_x1, finalmax_y1 = draw(img, XY_inliers2, XY_outliers2, line_X2, line_y_ransac2,
                                                              subplot4)
    finalcmin_x = min(finalmin_x, finalmin_x1)
    finalcmin_y = min(finalmin_y, finalmin_y1)
    finalcmax_x = max(finalmax_x, finalmax_x1)
    finalcmax_y = max(finalmax_y, finalmax_y1)
    subplot4.set_xlim([finalcmin_y - 10, finalcmax_y])
    subplot4.set_ylim([finalcmax_x + 10, finalcmin_x - 10])
    subplot4.title.set_text("RANSAC - No Filter")

    XY_inliers_final1, XY_outliers_final1 = filter(XY_inliers1,XY_outliers1)
    XY_inliers_final2, XY_outliers_final2 = filter(XY_inliers2,XY_outliers2)
    subplot5 = figure.add_subplot(1, 5, 5)
    img = np.zeros((img_shape[0], img_shape[1], 3))
    finalmin_x, finalmin_y, finalmax_x, finalmax_y = draw(img, XY_inliers_final1, XY_outliers_final1, line_X1,
                                                          line_y_ransac1, subplot5)
    finalmin_x1, finalmin_y1, finalmax_x1, finalmax_y1 = draw(img, XY_inliers_final2, XY_outliers_final2, line_X2,
                                                              line_y_ransac2, subplot5)
    finalcmin_x = min(finalmin_x, finalmin_x1)
    finalcmin_y = min(finalmin_y, finalmin_y1)
    finalcmax_x = max(finalmax_x, finalmax_x1)
    finalcmax_y = max(finalmax_y, finalmax_y1)
    subplot5.set_xlim([finalcmin_y - 10, finalcmax_y])
    subplot5.set_ylim([finalcmax_x + 10, finalcmin_x - 10])
    subplot5.title.set_text("RANSAC With Filter")
    plt.show()
def returnimage(img, XY_inliers, XY_outliers):
    for i in XY_inliers:
        img[int(i[0]), int(i[1]), 0] = 255

    for i in XY_outliers:
        img[int(i[0]), int(i[1]), 1] = 255
    return img


def test(img_path,img_output):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_shape = np.shape(img)
    iterr = 1
    edge = morphological_gradient(img, iterr)
    Top, Bottom = get_topbottom(edge)
    X1, y1, inlier_mask, outlier_mask, line_X1, line_y_ransac1= RANSAC(Top, img_shape)
    XY_inliers1, XY_outliers1, XY_real_ordered1 = findinliers(X1, y1, line_X1, line_y_ransac1)
    X2, y2, inlier_mask, outlier_mask, line_X2, line_y_ransac2 = RANSAC(Bottom, img_shape)
    XY_inliers2, XY_outliers2, XY_real_ordered2 = findinliers(X2, y2, line_X2, line_y_ransac2)
    XY_inliers_final1, XY_outliers_final1 = filter(XY_inliers1,XY_outliers1)
    XY_inliers_final2, XY_outliers_final2 = filter(XY_inliers2,XY_outliers2)
    img = np.zeros((img_shape[0], img_shape[1], 3))
    img = returnimage(img, XY_inliers_final1, XY_outliers_final1)
    img = returnimage(img, XY_inliers_final2, XY_outliers_final2)

    tt = os.path.basename(img_output)
    ttlen = len(tt)
    txtfile = img_output[:-ttlen]+"txt/" + tt[:-4] + ".txt"
    inliers_minx = np.amin(XY_inliers1, axis=0)[0]
    inliers_miny = np.amin(XY_inliers1, axis=0)[1]
    inliers_maxx = np.amax(XY_inliers1, axis=0)[0]
    inliers_maxy = np.amax(XY_inliers1, axis=0)[1]

    with open(txtfile, 'w') as f:
        f.write(f"{inliers_minx}, {inliers_miny}, {inliers_maxx}, {inliers_maxy}, {img_shape[0]}, {img_shape[1]}")

    inliers_minx = np.amin(XY_inliers2, axis=0)[0]
    inliers_miny = np.amin(XY_inliers2, axis=0)[1]
    inliers_maxx = np.amax(XY_inliers2, axis=0)[0]
    inliers_maxy = np.amax(XY_inliers2, axis=0)[1]

    with open(txtfile, 'a') as f:
        f.write("\n")
        f.write(f"{inliers_minx}, {inliers_miny}, {inliers_maxx}, {inliers_maxy}, {img_shape[0]}, {img_shape[1]}")

    line_X1 = [i[0] for i in line_X1]
    line_X2 = [i[0] for i in line_X2]
    line_y_ransac1 = [i[0] for i in line_y_ransac1]
    line_y_ransac2 = [i[0] for i in line_y_ransac2]
    data1 = np.column_stack([line_X1, line_y_ransac1])
    data2 = np.column_stack([line_X2, line_y_ransac2])
    datacombined = np.column_stack([data1, data2])


    csvfile = img_output[:-ttlen]+"csv/" + tt[:-4] + ".csv"
    fields = ['x1', 'y1', 'x2', 'y2']
    with open(csvfile, 'w') as n:
        write = csv.writer(n)
        write.writerow(fields)
        write.writerows(datacombined)

    cv2.imwrite(img_output,img)

    lw = 1
    plt.imshow(img)
    plt.plot(line_y_ransac1,
        line_X1,
        color="cornflowerblue",
        linewidth=lw)
    plt.plot(line_y_ransac2,
        line_X2,
        color="cornflowerblue",
        linewidth=lw)
    plt.xlim([0,img_shape[1]])
    plt.ylim([img_shape[1],0])
    Ransac_visualized = img_output[:-ttlen]+"Ransac_visualized/" + tt[:-4] + ".png"
    plt.savefig(Ransac_visualized)
    plt.close()

def Convert_folder(Input_directory,Output_directory):
    for entry in os.scandir(Input_directory):
        if entry.is_file() and entry.name[-4:]==".png":
            img_path = entry.path
            imgname = entry.name
            imgoutput = os.path.join(Output_directory,imgname)
            test(img_path,imgoutput)
def Convert_entired_irectory(Input_directory,Output_directory):
    for (root, dirs, files) in os.walk(Input_directory, topdown=True):
        if files != [] and dirs == []:
            for entry in os.scandir(root):
                if entry.name[-4:] == ".png":
                    img_path = entry.path
                    imgname = entry.name
                    output_mask_path = os.path.join(Output_directory, os.path.basename(root))
                    if os.path.isdir(output_mask_path) == False:
                        os.mkdir(output_mask_path)
                        os.mkdir(os.path.join(output_mask_path,"txt"))
                        os.mkdir(os.path.join(output_mask_path,"csv"))
                        os.mkdir(os.path.join(output_mask_path,"Ransac_visualized"))

                    imgexist = False
                    for entry in os.scandir(output_mask_path):
                        if entry.name == imgname:
                            imgexist = True
                    if imgexist == False:
                        print(imgname, os.path.basename(root))
                        imgoutput = os.path.join(output_mask_path, imgname)
                        test(img_path, imgoutput)

if __name__ == '__main__':
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/01-DEC-21_02-30AM_004_processed_02-30-54_733_015/0000079.png"
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/01-DEC-21_02-30AM_004_processed_02-30-54_733_015/0000078.png"
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/30-NOV-21_09-21PM_005_processed_21-21-44_493_003/0000000.png"
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/30-NOV-21_09-21PM_005_processed_21-21-44_493_003/0000000.png"
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/01-DEC-21_01-22AM_002_processed_01-22-48_857_254/0000092.png"
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/30-NOV-21_11-35PM_000_processed_22-54-05_183_008/0000091.png"
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/01-DEC-21_01-22AM_002_processed_01-22-48_857_258/0000000.png"
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks_separated/01-DEC-21_01-22AM_002_processed_01-22-48_857_251/0000056.png"
    # entirePipeline(img_path)
    # Input_directory = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/01-DEC-21_01-22AM_002_processed_01-22-48_857_251"
    # Output_directory = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks_separated/01-DEC-21_01-22AM_002_processed_01-22-48_857_251"
    # Convert_folder(Input_directory,Output_directory)

    Input_directory = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks"
    Output_directory = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks_separated"
    Convert_entired_irectory(Input_directory, Output_directory)

    #error happened in /Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks_separated/01-DEC-21_02-30AM_004_processed_02-30-54_733_054
    #investigation
    # img_path = "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks/01-DEC-21_02-30AM_004_processed_02-30-54_733_054/0000021.png"
    # entirePipeline(img_path)
    #problem fixed. the slope of the m line separting top and bottom halves was infinite becasue both the mid point x points were the same.

    #use lower minimum samples or higher residual threshold on the ball tip examples.

    #NEED TO FIX ERROR IN "/Users/zacham2/Desktop/StoneSmart/Coding/Trans2Seg/datasets/transparent/Nov_ModifiedLVEdata/masks_separated/30-NOV-21_11-35PM_000_processed_22-54-05_183_008"
    #INVESTIGATION

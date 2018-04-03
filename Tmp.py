import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
# from cvxopt import matrix, solvers
# from cvxopt.modeling import variable , op, dot
from scipy.optimize import leastsq

def MATMUL(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        print "Matrices are not compatible to Multiply. Check condition C1==R2"
        return

    # Create the result matrix
    # Dimensions would be rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]
    #print C

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    C = np.matrix(C).reshape(len(A),len(B[0]))

    return C


def func1(param, coordinates):
    addition = param[0]*coordinates[0]-param[1]*coordinates[1]+param[2]
    return addition

def func2(param, coordinates):
    addition = param[1]*coordinates[0]+param[0]*coordinates[1]+param[3]    
    return addition

def func(param,coordinates,coordinates1):
    addition = np.sqrt((func1(param,coordinates)-coordinates1[0])**2+(func2(param,coordinates)-coordinates1[1])**2)
    return addition

i=1
MIN_MATCH_COUNT = 10

fnm1='./Folder'+str(i)+'/'+str(1)+'.jpg'
fnm2='./Folder'+str(i)+'/'+str(2)+'.jpg'

img1 = cv2.imread(fnm1,0)          # queryImage
img2 = cv2.imread(fnm2,0) 		   # trainImage
# Initiate SIFT detector

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    size = len(src_pts)
    penalty_score = np.full(size, 0)
    has_been_selected = np.full(size,0) 
    quality = 0.0
    feature_points_src = []
    feature_points_dst = []
    fl = 0

    while fl == 0 :
		seed_index = random.randint(0,size-1)
		if has_been_selected[seed_index] == 0 and np.mean(penalty_score) >= penalty_score[seed_index]:    			
			has_been_selected[seed_index] = 1
			fl=1
			penalty_score = penalty_score + 1

    dist_list = []
    for i in range(size):
        diff_x = src_pts[i][0][0]-src_pts[seed_index][0][0]
        diff_y = src_pts[i][0][1]-src_pts[seed_index][0][1]
        dist = diff_x*diff_x + diff_y*diff_y
        dist_list.append([i,dist])
    dist_list.sort(key=lambda x: x[1])
    for i in range(5):
    	feature_points_src.append(src_pts[dist_list[i][0]])
    	feature_points_dst.append(dst_pts[dist_list[i][0]])
    i+=1
    while quality<=0.01:
        feature_points_src.append(src_pts[dist_list[i][0]])
        feature_points_dst.append(dst_pts[dist_list[i][0]])
        feature_points_src1 = np.float32(feature_points_src)
        feature_points_dst1 = np.float32(feature_points_dst)
        H, mask = cv2.findHomography(feature_points_src1, feature_points_dst1, cv2.RANSAC,5.0)
        i+=1
        h,w = np.shape(img1)
        pts_src = np.array([[0,0,h-1,h-1], [0, w-1,0,w-1], [1,1,1,1]])
        pts_prime = MATMUL(H,pts_src)

        pts_prime1 = [[],[]]
        pts_prime1[0] = pts_prime[0]/pts_prime[2]
        pts_prime1[1] = pts_prime[1]/pts_prime[2]
        pts_prime1 = np.array(pts_prime1)
        
        x = np.array([[0,0,h-1,h-1],[0,w-1,0,w-1]])
        y = np.array(pts_prime1.reshape(2,4))
        param_init = [1,2,3,4]
        params, success = leastsq(func,param_init,args=(x,y))
        
        H_ = np.array([[params[0],-params[1],params[2]], [params[1],params[0],params[3]], [0,0,1]])
        pts_ = MATMUL(H_,pts_src)
        Hy=pts_[:2,:]
        quality = (np.sum(np.sqrt(np.sum(np.square(y-Hy),axis=0))))/(h*w)
        print i,quality
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
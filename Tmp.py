import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
# from cvxopt import matrix, solvers
# from cvxopt.modeling import variable , op, dot
from scipy.optimize import leastsq
import scipy.misc
import maxflow
from numpy import concatenate, ones, zeros

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

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def graph_cut(I,src_pts):
    g = maxflow.Graph[float]()
    i_inf = np.inf
    nodeids = g.add_grid_nodes(I.shape)
    h,w = np.shape(I)
    mu, sigma = 0, 1.5 # mean and standard deviation
    weights = np.zeros((h,w))
    l=0
    for k in src_pts:
        weights_x = np.zeros(h)
        weights_y = np.zeros(w)
        weights_tmp = np.zeros((h,w))
        for i in range(h):
            weights_x[i]+=(k[0][0]-i)**2
        weights_x = np.tile(weights_x,(w,1))
        weights_x = np.transpose(weights_x)
        for i in range(w):    
            weights_y[i]+=(k[0][1]-i)**2
        weights_y = np.tile(weights_y,(h,1))
        weights_tmp = weights_x+weights_y
        weights_tmp = gaussian(np.sqrt(weights_tmp),np.mean(weights_tmp),np.std(weights_tmp))
        weights+=weights_tmp
    weights = 1.0/(weights+0.01)
    for i in range(h):
        for j in range(w):
            weights[i][j]=weights[i][j]*I[i][j]
    structure = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    g.add_grid_edges(nodeids, weights, structure=structure,symmetric=True)
    left_most = concatenate((np.arange(I.shape[0]).reshape(1, I.shape[0]), zeros((1, I.shape[0])))).astype(np.uint64)
    left_most = np.ravel_multi_index(left_most, I.shape)
    g.add_grid_tedges(left_most, i_inf, 0)

    
    right_most = concatenate((np.arange(I.shape[0]).reshape(1, I.shape[0]), ones((1, I.shape[0])) * (np.size(I, 1) - 1))).astype(np.uint64)
    right_most = np.ravel_multi_index(right_most, I.shape)
    g.add_grid_tedges(right_most, 0, i_inf)
    x = g.maxflow()
    return x

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
    size_dst = len(dst_pts)
    penalty_score = np.full(size, 0)
    has_been_selected = np.full(size,0)
    align_quality = np.inf
    best_alignment_so_far = np.inf
    best_H = np.zeros((3,3))
    while align_quality>=100:
        quality = 0.0
        feature_points_src = []
        feature_points_dst = []
        fl = 0
        src_min_overlap_point_x = np.inf
        src_max_overlap_point_x = -np.inf
        dst_min_overlap_point_x = np.inf
        dst_max_overlap_point_x = -np.inf
        while fl == 0 :
    		seed_index = random.randint(0,size-1)
    		if has_been_selected[seed_index] == 0 and np.mean(penalty_score) >= penalty_score[seed_index]:    			
    			has_been_selected[seed_index] = 1
    			fl=1
    			penalty_score = penalty_score + 1

        dist_list = []
        for i in range(size):
            if src_min_overlap_point_x > src_pts[i][0][0]:
                src_min_overlap_point_x = src_pts[i][0][0]
            if src_max_overlap_point_x < src_pts[i][0][0]:
                src_max_overlap_point_x = src_pts[i][0][0]
            diff_x = src_pts[i][0][0]-src_pts[seed_index][0][0]
            diff_y = src_pts[i][0][1]-src_pts[seed_index][0][1]
            dist = diff_x*diff_x + diff_y*diff_y
            dist_list.append([i,dist])

        for i in range(size_dst):
            if dst_min_overlap_point_x > dst_pts[i][0][0]:
                dst_min_overlap_point_x = dst_pts[i][0][0]
            if dst_max_overlap_point_x < dst_pts[i][0][0]:
                dst_max_overlap_point_x = dst_pts[i][0][0]
        dist_list.sort(key=lambda x: x[1])

        resize_img1 = img1[int(src_min_overlap_point_x):int(src_max_overlap_point_x),:]
        resize_img2 = img2[int(dst_min_overlap_point_x):int(dst_max_overlap_point_x),:]
        resize_img1 = cv2.resize(resize_img1, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
        resize_h,resize_w = np.shape(resize_img1)
        resize_img2 = cv2.resize(resize_img2, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
        resize_img2 = cv2.resize(resize_img2, (resize_w, resize_h) ,interpolation = cv2.INTER_CUBIC)

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
        edges1 = cv2.Canny(resize_img1,100,200)
        warped_image = cv2.warpPerspective(resize_img2,H,(np.size(resize_img2,1),np.size(resize_img2,0)))
        edges2 = cv2.Canny(warped_image,100,200)
        edge_map = np.abs(edges1-edges2)
        print edge_map
        align_quality = graph_cut(edge_map,src_pts)
        print align_quality, np.mean(penalty_score)
        if best_alignment_so_far > align_quality:
            best_alignment_so_far = align_quality
            best_H = H
        if align_quality < 100:
            break
        if np.mean(penalty_score) >= 2:
            break
    print H
    

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
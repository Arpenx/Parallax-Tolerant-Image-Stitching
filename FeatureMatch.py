import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from cvxopt import matrix, solvers
from cvxopt.modeling import variable , op, dot

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
    for i in range(9):
    	feature_points_src.append(src_pts[dist_list[i][0]])
    	feature_points_dst.append(dst_pts[dist_list[i][0]])
    i+=1
    while quality<=0.01:
        feature_points_src.append(src_pts[dist_list[i][0]])
        feature_points_dst.append(dst_pts[dist_list[i][0]])
        feature_points_src = np.float32(feature_points_src)
        feature_points_dst = np.float32(feature_points_dst)
        H, mask = cv2.findHomography(feature_points_src, feature_points_dst, cv2.RANSAC,5.0)
        i+=1
        h,w = np.shape(img1)
        pts_src = np.array([[0,0,h-1,h-1], [0, w-1,0,w-1], [1,1,1,1]])
        pts_prime = MATMUL(H,pts_src)
        a = variable()
        b = variable()
        c = variable()
        d = variable()
        print pts_src
        print pts_prime
        
        pts_prime1 = [[],[]]
        pts_prime1[0] = pts_prime[0]/pts_prime[2]
        pts_prime1[1] = pts_prime[1]/pts_prime[2]
        pts_prime1 = np.array(pts_prime1)
        pts_prime1 = pts_prime1.transpose()
        pts_arg1 = MATMUL(H,pts_src)
        """print pts_prime[1][0], pts_prime
        pts_arg1 = c + d - pts_prime[0][0][0] 
        			- pts_prime[0][0][1] - b*(w-1)
        			+ c + a*(w-1) + d - pts_prime[1][0][0]
        			- pts_prime[1][0][1] + a*(h-1) + c + b*(h-1)
        			+ d - pts_prime[2][0][0] - pts_prime[2][0][1]
        			+ a*(h-1) - b*(w-1) + c + b*(h-1) + a*(w-1)
        			+ d - pts_prime[3][0][0] - pts_prime[3][0][1]"""
        #lp = solvers.lp(pts_arg1-pts_prime)
        #lp.solve()
        #print a.value
        quality = 2

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)


plt.imshow(img3, 'gray'),plt.show()

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
edges1 = cv2.Canny(img1,100,200)
plt.imshow(edges1,cmap = 'gray'), plt.show()

edges2 = cv2.Canny(img2,100,200)
plt.imshow(edges2,cmap = 'gray'), plt.show()

#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
#plt.imshow(img3,),plt.show()
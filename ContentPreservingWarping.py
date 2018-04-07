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
fnm1='./Folder'+str(i)+'/'+str(1)+'.jpg'
fnm2='./Folder'+str(i)+'/'+str(2)+'.jpg'

img1 = cv2.imread(fnm1,0)          # queryImage
img2 = cv2.imread(fnm2,0) 		   # trainImage

I=np.asarray(img1)
I_=np.asarray(img2)

grid=np.asarray(I.shape)

grid[0]=grid[0]/100
grid[1]=grid[1]/250

feature_pts=[[34,67,1],[100,303,1],[54,101,1]]
H=[[10,0.5,20],[0.8,2,50],[1.5,2,1]]

for pts in feature_pts:
	Pts_Grid=np.asarray([[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1]])

	Pts_Grid[0][0]=(pts[0]/grid[0])*grid[0]
	Pts_Grid[0][1]=(pts[1]/grid[1])*grid[1]

	Pts_Grid[1][0]=(pts[0]/grid[0] + 1)*grid[0]
	Pts_Grid[1][1]=(pts[1]/grid[1])*grid[1]

	Pts_Grid[2][0]=(pts[0]/grid[0])*grid[0]
	Pts_Grid[2][1]=(pts[1]/grid[1] + 1)*grid[1]

	Pts_Grid[3][0]=(pts[0]/grid[0] + 1)*grid[0]
	Pts_Grid[3][1]=(pts[1]/grid[1] + 1)*grid[1]

	Pts_Grid=MATMUL(H,np.transpose(Pts_Grid))
	pts_=MATMUL(H,np.transpose([pts]))

	x=pts_[0]
	y=pts_[1]

	Tmp=np.asarray([[1,-1,-1,1],[-1,1,0,0],[-1,0,1,0],[1,0,0,0]])
	Pts_Grid=np.asarray(Pts_Grid[:-1][:])

	# print Pts_Grid
	
	Alphabets=np.array(MATMUL(Tmp,np.transpose(Pts_Grid)))

	a=Alphabets[0][0]
	b=Alphabets[1][0]
	c=Alphabets[2][0]
	d=Alphabets[3][0]
	e=Alphabets[0][1]
	f=Alphabets[1][1]
	g=Alphabets[2][1]
	h=Alphabets[3][1]

	# print Alphabets

	A = a*f - b*e
	B = e*x - a*y + a*h - d*e + c*f - b*g
	C = g*x - c*y + c*h -d*g

	D = a*g - c*e
	E = e*x - a*y + a*h - d*e - c*f + b*g
	F = f*x - b*y + b*h -d*f

	# print A,B,C,D,E,F

	u=np.array([-1.0,-1.0])
	v=np.array([-1.0,-1.0])

	if(A):
		u[0]=(-B+np.sqrt(B*B-4*A*C))/(2*A)
		u[1]=(-B-np.sqrt(B*B-4*A*C))/(2*A)

		v[0]= (x - b*u[0] - d)/(a*u[0] +c)
		v[1]= (x - b*u[1] - d)/(a*u[1] +c)
	else:
		u[0]=-C/B

		v[0]= -F/E

	coef=[(1-u[0])*(1-v[0]),(1-u[0])*(v[0]),(u[0])*(1-v[0]),(u[0])*(v[0])]
	
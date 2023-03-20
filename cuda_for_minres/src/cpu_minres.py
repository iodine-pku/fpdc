import numpy as np
import time
from scipy.sparse import dia_matrix

total =128*128*128
N = 128

#make sparse matrix
data = 26*np.ones(total)
offsets = np.array([0])
A = dia_matrix((data,offsets),shape=(total,total),dtype="float64")
A = A.tolil()
for i in range(N):
    for j in range(N):
        for k in range(N):
            for bz in range(-1,2):
                for by in range(-1,2):
                    for bx in range(-1,2):
                        if i+bz>=0 and i+bz < N and j + by>=0 and j+by<N and k+bx>=0 and k+bx<N and bx*bx+by*by+bz*bz!=0:
                            A[i*N*N+ j*N+k,(i+bz)*N*N+ (j+by)*N+(k+bx)] = -1   
A = A.tocsr()
x_exact = np.ones(total,dtype='float64')
x0 = np.zeros(total,dtype='float64')
b = A @ x_exact

#algorithm
def cpu_minres(A,x0,b,maxit):
    x = np.array(x0)
    r = b - A @ x0
    p0 = np.array(r)
    s0 = A @ p0
    p1 = np.array(p0)
    s1 = np.array(s0)
    for iter in range(1,maxit):
        p2 = np.ndarray.copy(p1)
        p1 = np.ndarray.copy(p0)
        s2 = np.ndarray.copy(s1)
        s1 = np.ndarray.copy(s0)
        alpha = np.dot(r,s1)/np.dot(s1,s1)
        x = x + alpha * p1
        r = r - alpha * s1
        p0 = np.ndarray.copy(s1)
        s0 = A @ s1
        beta1 = np.dot(s0,s1)/np.dot(s1,s1)
        p0 = p0 - beta1* p1
        s0 = s0 - beta1* s1
        if iter>1:
            beta2 = np.dot(s0,s2)/np.dot(s2,s2)
            p0 = p0 - beta2* p2
            s0 = s0 - beta2* s2
        #print("iter{} finished!,x0={},beta1={}".format(iter,x[0],beta1))
            
    return x, r

st = time.time()
[x,r]= cpu_minres(A,x0,b,50)
sp = time.time()-st
relative_error = np.sqrt(np.sum(np.square(x-x_exact)))/np.sqrt(np.sum(np.square(x_exact)))
residue = np.sqrt(np.sum(np.square(r)))
print("relative error={}, residue={}".format(relative_error,residue))
print("elapsed time:{} ".format(sp))
import numpy as np
import pylab as plt
from mpl_toolkits import mplot3d
from copy import deepcopy
B=deepcopy(A)

def dist(A,C):
    x=0
    for i in range(N):
        for j in range(N):
            x+=(A[i,j]-C[i,j,])**2
    return x


def proj2(Z):
    plt.subplot(1,3,1)
    T1=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            T1[i,j]=max([Z[i,j,k] for k in range(N)])

    plt.imshow(T1,cmap="binary",interpolation="none")
    plt.subplot(1,3,2)
    T2=np.zeros((N,N))
    for j in range(N):
        for k in range(N):
            T2[j,k]=max([Z[i,j,k] for i in range(N)])

    plt.imshow(T2,cmap="binary",interpolation="none")
    plt.subplot(1,3,3)
    T3=np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            T3[i,k]=max([Z[i,j,k] for j in range(N)])

    plt.imshow(T3,cmap="binary",interpolation="none")
    return (T1,T2,T3)
    plt.show()
M=1000
L=[]
G=[]
S=[k/(8*M) for k in range(M)]
for k in range(M):
    if k%int(M/10)==0:
        print(int(10-10*k/M))
    s=S[k]
    B=deepcopy(A)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if B[i,j,k]>=s:
                    B[i,j,k]=1
                else:
                    B[i,j,k]=0
    P1,P2,P3=proj2(B)
    d=dist(P1,f1)+dist(P2,f2)+dist(P3,f3)
    L.append(d)
    G.append(B)

for k in range(M):
    if L[k]==min(L):
        X=deepcopy(G[k])


proj2(X)

def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax
ax = make_ax(True)
ax.voxels(X, facecolors='#1f77b430', edgecolors='gray', shade=False)
plt.show()
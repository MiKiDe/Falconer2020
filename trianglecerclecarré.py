import numpy as np
import pylab as plt
from mpl_toolkits import mplot3d

N=30

f1=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if np.sqrt((i-N/2)**2+(j-N/2)**2)<=N/3:
            f1[i,j]=1

f2=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if N/3<=i<=2*N/3 and N/3<=j<=2*N/3:
            f2[i,j]=1

f3=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i+N/4>=N/3  and j<=3*N/3-(i+N/4)/2 and j>=(i+N/4)/2:
            f3[i,j]=1

def proj1(A):
    plt.figure()
    T=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            T[i,j]=sum([A[i,j,k] for k in range(N)])

    plt.imshow(T,cmap="binary",interpolation="none")
    plt.show()

def proj2(A):
    plt.figure()
    T=np.zeros((N,N))
    for j in range(N):
        for k in range(N):
            T[j,k]=sum([A[i,j,k] for i in range(N)])

    plt.imshow(T,cmap="binary",interpolation="none")
    plt.show()

def proj3(A):
    plt.figure()
    T=np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            T[i,k]=sum([A[i,j,k] for j in range(N)])

    plt.imshow(T,cmap="binary",interpolation="none")
    plt.show()
A=np.zeros((N,N,N))
R=10
def f(X):
    s=0
    for x in X:
        s+=x
    return 1-np.exp(-R*s)
def g(X):
    return 1-f(X)
def F(M):
    s=0
    for i in range(N):
        for j in range(N):
            s+=(f1[i,j]-f([A[i,j,k] for k in range(N)]))**2
    for j in range(N):
        for k in range(N):
            s+=(f2[j,k]-f([A[i,j,k] for i in range(N)]))**2
    for i in range(N):
        for k in range(N):
            s+=(f3[i,k]-f([A[i,j,k] for j in range(N)]))**2
    return s
e=F(A)
t=10**(-3)
def gradF(i,j,k,A):
    return 2*R*((g([A[i,j,k] for k in range(N)])*(f1[i,j]-f([A[i,j,k] for k in range(N)]))+g([A[i,j,k] for i in range(N)])*(f2[j,k]-f([A[i,j,k] for i in range(N)]))+g([A[i,j,k] for j in range(N)])*(f3[i,k]-f([A[i,j,k] for j in range(N)]))))
d=1
while d>0 and F(A)>80:
    q=F(A)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                v=A[i,j,k]
                m=gradF(i,j,k,A)
                A[i,j,k]=max(0,min(v+t*m,1))
    d=q-F(A)

    print(q)



proj1(A)
proj2(A)
proj3(A)
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax
ax = make_ax(True)
ax.voxels(A, facecolors='#1f77b430', edgecolors='gray', shade=True)
plt.show()
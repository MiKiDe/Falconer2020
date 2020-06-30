import numpy as np
import pylab as plt
from mpl_toolkits import mplot3d
import numpy.random as rd

N=50

## Choix des figures
# Formes géométriques : rond, carré, triangle
g1=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if np.sqrt((i-N/2)**2+(j-N/2)**2)<=N/4:
            g1[i,j]=1

g2=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if N/3<=i<=2*N/3 and N/3<=j<=2*N/3:
            g2[i,j]=1

g3=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i+N/4>=3*N/5  and j<=3*N/3-(i+N/4)/2 and j>=(i+N/4)/2:
            g3[i,j]=1

# Chiffres : 1, 2 et 3
c1=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if j==int(N/2) and N/4<i<3*N/4:
            c1[i,j]=1
        if j==int(N/2)-1  and N/4<i<3*N/4:
            c1[i,j]=1

c2=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i==int(N/4) and N/3<j<2*N/3:
            c2[i,j]=1
        if j==int(2*N/3) and N/4<i<N/2:
            c2[i,j]=1
        if i==int(N/2) and N/3<j<2*N/3:
            c2[i,j]=1
        if j==int(N/3) and N/2<i<3*N/4:
            c2[i,j]=1
        if i==int(3*N/4) and N/3<j<2*N/3:
            c2[i,j]=1
c3=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i==int(N/4) and N/3<j<2*N/3:
            c3[i,j]=1
        if j==int(2*N/3) and N/4<i<N/2:
            c3[i,j]=1
        if i==int(N/2) and N/3<j<2*N/3:
            c3[i,j]=1
        if j==int(2*N/3) and N/2<i<3*N/4:
            c3[i,j]=1
        if i==int(3*N/4) and N/3<j<2*N/3:
            c3[i,j]=1

# Fixer les formes pour la suite :
f1 = c1
f2 = c2
f3 = c3
## Fonction de projection et de figures
def proj2(Z):
    plt.subplot(1,3,1)
    T1=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            T1[i,j]=max(Z[i,j,:])

    plt.imshow(T1,cmap="binary",interpolation="none")
    plt.subplot(1,3,2)
    T2=np.zeros((N,N))
    for j in range(N):
        for k in range(N):
            T2[j,k]=max(Z[:,j,k])

    plt.imshow(T2,cmap="binary",interpolation="none")
    plt.subplot(1,3,3)
    T3=np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            T3[i,k]=max(Z[i,:,k])
    plt.imshow(T3,cmap="binary",interpolation="none")
    #return (T1,T2,T3)
    plt.show()

def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

## Condition initiale :
A=rd.random((N,N,N))
A*=0.15

## Choix de la fonction coût :
p=np.log(N)/np.log(1.1)#pour avoir norme inf<=norme p<= 1.1*norme inf
def f(X):
    s=0
    for x in X:
        s+=x**p
    return s**(1/p)
def F(M):
    s=0
    for i in range(N):
        for j in range(N):
            s+=(f1[i,j]-f(A[i,j,:]))**2
    for j in range(N):
        for k in range(N):
            s+=(f2[j,k]-f(A[:,j,k]))**2
    for i in range(N):
        for k in range(N):
            s+=(f3[i,k]-f(A[i,:,k]))**2
    return s
e=F(A)
t=1
def derf(k,X):
    if f(X)==0:
        return 0
    else:
        return (X[k]**(p-1))/(f(X)**(p-1))
def gradF(i,j,k,A):
        return-2*derf(k,A[i,j,:])*(f1[i,j]-f(A[i,j,:]))-2*derf(i,A[:,j,k])*(f2[j,k]-f(A[:,j,k]))-2*derf(j,A[i,:,k])*(f3[i,k]-f(A[i,:,k]))

## Boucle principale : descente de gradient
s = 50 # seuil au deça duquel on arrête la boucle
d=10 # Pas de la descente de gradient

while q>s:
# On choisit de faire des pas de descente plus petit à mesure qu'on se rapproche d'un minimum :
    if d<0.5:
        t=10**(-3)
    if d<1:
        t=10**(-2)
    if F(A)<300:
        t=10**(-1)

    q=F(A)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                v=A[i,j,k]
                m=gradF(i,j,k,A)
                A[i,j,k]=max(0,min(v-t*m,1))
    d=q-F(A)

    print(q)



# Figures :
proj2(A)
ax = make_ax(True)
ax.voxels(A, facecolors='#1f77b430', edgecolors='gray', shade=True)
plt.show()

## Seuil

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

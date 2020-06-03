import numpy as np
import pylab as plt
import pyqtgraph as pg
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
import tikzplotlib
import math
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.collections import LineCollection

## Paramètres
s_eps = lambda eps,L,m : int(L*(1/2 + 1/(m-1))*np.pi/(4*eps)) + 1 #s en fonction de epsilon, L et m

eta = epsilon
nbs = lambda L,delta,eta, beta : abs(int(L*np.sin(delta + beta)/(eta*np.sin(delta))))+1 #nombre de segments parallèles pour remplacer ceux de l'étape 1
## Fonctions utiles
#Longueur d'un intervalle:
def longJ(J):
    #print(J)
    a=J[1]
    b=J[0]
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
#Angle entre deux vecteurs:
def angle(x,y):
    unit_vector_1 = x / np.linalg.norm(x)
    unit_vector_2 = y / np.linalg.norm(y)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)

## Transformations

#Transformation T de la prop 1, valable pour l'intervalle [0,1] seulement pour le moment

def T(J,m,alpha):
    e1 = np.array([1,0])
    A = np.zeros((len(J)*m,2,2))
    R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
    n = len(J)
    for i,j in enumerate(J):
        for k in range(m):
            A[i*m+k,0]=(1/(m*np.cos(alpha)))*np.dot(R,j[0]) + k/m*e1
            A[i*m+k,1]=(1/(m*np.cos(alpha)))*np.dot(R,j[1]) + k/m*e1
    return A

def Trans1(J,s,r,m):
    alpha=np.pi/(4*s)
    if r>0:
        return Trans1(T(J,m,alpha),s,r-1,m)
    else: return J
I = np.array([[[0,0],[1,0]]])
J = Trans1(I,5,5,2)
for u in J:
    plt.plot([u[0,0],u[1,0]],[u[0,1],u[1,1]],'r')
#plt.show()

# Transformation T2 de la prop 2:

def T2(delta, eta, J):
    a=J[1]
    b=J[0]
    v = J[1]-J[0]
    l = longJ(J)
    l = round(l,7)
    beta=np.arctan((b[1]-a[1])/(b[0]-a[0]))
    beta = round(beta,7)
    k = nbs(l,delta,eta,beta)
    O = J[0]
    F = np.zeros((k+1,2,2))
    for i in range(0,k+1):
        u = O+i*v/k
        u1 = u + np.array([eta/2,0])
        u2 = u + np.array([-eta/2,0])
        #plt.plot([u1[0],u2[0]],[u1[1],u2[1]],'g-')
        F[i,0],F[i,1] = u1,u2
    return F

def Trans2(delta,eta,J):
    a,b = J[0]
    beta=-np.arctan((a[1]-b[1])/(b[0]-a[0]))
    beta = round(beta,7)
    l = longJ(J[0])
    l = round(l,7)
    n = len(J)
    k = nbs(l,delta,eta,beta)
    k = k+1
    A = np.zeros((k*n,2,2))
    for i in range(len(J)):
        F = T2(delta,eta,J[i])
        A[i*k:(i+1)*k] = F
    return A

B = Trans2(0.1,0.1,J)
for u in B:
    plt.plot([u[0,0],u[1,0]],[u[0,1],u[1,1]],'g')
#plt.show()
# Transformation affine de la prop 3:

def U(I,alpha,delta,mu): # A faire fonctionner pour I neq [0,1] !
    lambda1, lambda2 = -1/(np.sqrt(2)*np.cos(alpha+delta)),1/(np.sqrt(2)*np.cos(alpha))
    u_1 = (np.array([[-1,1]])/np.sqrt(2))
    u_2 = np.array([[1,1]])/np.sqrt(2)
    v_1 = np.array([[np.cos(alpha+delta),-np.sin(alpha+delta)]])
    v_2 = np.array([[np.cos(alpha),-np.sin(alpha)]])
    C = lambda1*np.dot(np.transpose(v_1),u_1) + lambda2*np.dot(np.transpose(v_2),u_2)
    A = np.linalg.inv(np.transpose(C))
    e2 = np.array([0,1])
    b = mu*(e2-np.dot(A,e2))
    return np.dot(A,I[0])+b,np.dot(A,I[1])+b

def Trans3(J,alpha,delta,mu):
    A = np.zeros(np.shape(J))
    for i,j in enumerate(J):
        A[i] = U(j,alpha,delta,mu)
    return A

C = Trans3(B,0.1,0.01,0)
for u in C:
    plt.plot([u[0,0],u[1,0]],[u[0,1],u[1,1]],'b')
plt.show()

def sym(J,mu):
    return np.array([[J[0,0],2*mu-J[0,1]],[J[1,0],2*mu-J[1,1]]])
    #return [np.array([I[0][0],mu-I[0][1]]),np.array([I[1][0],mu-I[1][1]])]

## Application de toutes les étapes :
# Améliorations possibles : faire les modifs en place, stocker qu'une partie de la structure
def E(I,eps, delta, rho, alpha):
    def E4(J,delta,s,m,eta):
        l = longJ(J[0])
        C = np.zeros((0,2,2))
        for i,j in enumerate(J):
            mu = j[0,1]
            jj = np.array([j])
            Aj = Trans1(jj,s,s,m)
            F = Trans2(delta,eta,jj)
            F = Trans3(F,alpha,delta,mu)
            for f in range(len(F)):
                F[f] = sym(F[f],mu)
            C = np.concatenate((C,F))
        return C
    L = longJ(I[0])
    m = 2
    s = s_eps(eps/4,L,m)
    print(s)
    J = Trans1(I,s,s,m)
    print("1 OK")
    eta = eps/(4*m**s)
    J = Trans2(delta,eta,J)
    print("2 OK")
    J = Trans3(J,delta,alpha,0)
    print("3 OK")
    s = s_eps(eps/4,L,m)
    print(s)
    J = E4(J,delta,s,m,eta)
    print("4 OK")
    print(np.shape(J))
    return J

I = np.array([[[0,0],[1,0]]])
eps = 0.5
delta = np.pi/10
rho = 1e-2
alpha = np.pi/10

A = E(I,eps,delta,rho,alpha)

## Calcul enveloppe convexe
def envconv(E):
    def points(E):
        pts = []
        for e in E:
            pts.append(e[0])
            pts.append(e[1])
        return np.array(pts)
    pts = points(E)
    hull = ConvexHull(pts)
    return pts,hull

# Pour plot :

pts, hull = envconv(A)




## Calcul de la distribution de la projection
theta = np.pi/8

def reord(D): #réarrange les segments de sorte que l'extrémité gauche soit à gauche (a implémenter directement dans les algos précédents
    k = 0
    for i,j in enumerate(D):
        a,b = np.copy(j[0]),np.copy(j[1])
        if j[0,0]>j[1,0]:
            D[i,1],D[i,0] = a,b
            k+=1
    print(k)

def p(x,theta): #calcul du vecteur projeté sur la droite l_theta
    utheta = np.array([-np.sin(theta),np.cos(theta)])
    xet = np.dot(x,utheta)
    return xet*utheta

def tri(E): #tri fusion des intervalles par ordonnées croissante de l'ext. g.
    def fusion(t1,t2):
        i1,i2,n1,n2=0,0,len(t1),len(t2)
        t=np.zeros((n1+n2,2,2))
        i = 0
        while i1<n1 and i2<n2:
            if t1[i1,0,0]<t2[i2,0,0]:
                t[i] = t1[i1]
                i1+=1
                i+=1
            else:
                t[i] = t2[i2]
                i2+=1
                i+=1
        if i1==n1:
            t[i1+i2:] = t2[i2:]
        else:
            t[i1+i2:] = t1[i1:]
        return t
    def tri(t):
        n=len(t)
        if n<2:
            return t
        else:
            m=n//2
            return fusion(tri(t[:m]),tri(t[m:]))
    return tri(E)

def proj(theta,Ee): #calcul de la distribution de projection, en O(nlogn) où n est le nombre de segments
    #Etape 1 : calcul des intervalles projetés
    P = np.zeros(np.shape(Ee))
    for i,j in enumerate(Ee):
        P[i] = [p(j[0],theta),p(j[1],theta)]
    #Etape 2 : tri de la liste des intervalles par ordonnées croissante de l'extrémité gauche
    reord(P)
    P = tri(P)
    #Etape 3 : on parcourt la liste des projections en concaténant si besoin :
    def superposition(A,B): #vérifie si A et B deux intervalles se superposent
        if A[1,0]>= B[0,0]:
            return True
        else:
            return False
    def concatenation(A,B):
        return np.array([A[0],B[1]])
    n = len(P)
    T = np.zeros((n,2,2))
    T[0] = P[0]
    i,j = 0,1
    while j<n:
        if superposition(T[i],P[j]):
            T[i] = concatenation(T[i],P[j])
        else:
            i+=1
            T[i] = P[j]
        j+=1
    D = T[:(i+1)]
    return D

D = proj(0.1,A)

def testtri(D):
    B = True
    k = 0
    n = len(D)
    for i in range(n-1):
        if D[i,0,0]>D[i+1,0,0]:
            B = False
            break
    return B

def testsupp(D):
    B = True
    k=0
    n = len(D)
    for i in range(n-1):
        k = i
        if D[i,1,0]>D[i+1,0,0]:
            B = False
            break
    return B, k
## Plot
plotWidget = pg.plot(title="Test")

t = time.time()
X = []
Y = []

for u in A:
    X.append(u[0,0])
    X.append(u[1,0])
    Y.append(u[0,1])
    Y.append(u[1,1])
plotWidget.plot(X,Y, pen='g')

X = []
Y = []

for simplex in hull.vertices:
    X.append(pts[simplex, 0])
    Y.append(pts[simplex, 1])
plotWidget.plot(X,Y, pen = 'w')

def relier(a,b,couleur='gray'):
    plt.plot([a[0],b[0]],[a[1],b[1]],color=couleur)

for v in hull.vertices:
    a = pts[v]
    b = p(pts[v],theta)
    #plotWidget.plot([a[0],b[0]],[a[1],b[1]], pen='y')

# for u in D:
#     plotWidget.plot([u[0,0],u[1,0]],[u[0,1],u[1,1]],pen = 'r')

print(time.time()-t)

##
fig, ax = plt.subplots()
Fset = LineCollection(A)
Pr = LineCollection(D, colors = ['r'])
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2,1)
ax.add_collection(Fset)
ax.add_collection(Pr,'r')
plt.show()

## Vérification de la densité :

# A faire : regarder la densité de la projection.

def densite(D): #D doit être la distribution de projection, sans doublon
    #On calcule le plus petit intervalle contenant toutes les projections
    L = np.linalg.norm(D[0,0]-D[-1,1])
    s = 0
    for j in D:
        l = np.linalg.norm(j[0]-j[1])
        s+=l
    return s/L
densite(D)


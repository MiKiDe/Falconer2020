import numpy as np
import pylab as plt
#on code un segment par le couple de ses deux extrémités
def T(J,s,m):  #renvoie la liste des nouveaux segments
    alpha=np.pi/(4*s)
    a=J[0]
    b=J[1]
    l=np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)    #longuer de J
    L=[a+(k/m)*(b-a) for k in range(1,m+1)]     #liste des endroits d'où on fait partir les nouveaux segments
    beta=np.arctan((a[1]-b[1])/(b[0]-a[0]))   #angle que fait J avec l'horizontale
    v=(l/(m*np.cos(alpha)))*np.array([-np.cos(alpha+beta),np.sin(alpha+beta)])  #vecteur directeur des nouveaux segments
    
    return [(L[k]+v,L[k]) for k in range(m)]

def E(J,s,m):
    J=[J]
    for q in range(s+1):
        A=[]
        for k in range(len(J)):
            u=J[k]
            plt.plot([u[0][0],u[1][0]],[u[0][1],u[1][1]])
            R=T(u,s,m)
            for i in range(m):          #on construit la suite des nouveaux segments a chaque étape
                A.append(R[i])
            
        J=A
    return J
    
    
I=(np.array([1/2,1/2]),np.array([1,1/2]))

E(I,1,3)

plt.text(0.73,0.46,r'$J$',fontsize="15")

plt.axis('square')
plt.axis([0.4,1.1,0.3,0.8])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

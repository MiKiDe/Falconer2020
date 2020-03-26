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
    copie=J[0]
    J=[J]
    for q in range(s+1):
        A=[]
        for k in range(len(J)):
            u=J[k]
            if k==0:
                plt.plot([u[0][0],u[1][0]],[u[0][1],u[1][1]],color='k')
                w=u[0]
                v=w-copie
                w=copie+100*v
                plt.plot([w[0],copie[0]],[w[1],copie[1]],color='r')
                copie=u[0]
               
            R=T(u,s,m)
            for i in range(m):          #on construit la suite des nouveaux segments a chaque étape
                A.append(R[i])
            
        J=A
    return J
    
    
I=(np.array([1/2,1/2]),np.array([1,0.45]))

E(I,5,2)

plt.text(0.63,0.488,r'$E_r$',fontsize="15")
plt.text(0.6,0.52,r'$E_{r+1}$',fontsize="15")
plt.text(0.56,0.54,r'$E_{r+k}$',fontsize="15")
plt.text(0.49,0.52,r'$u_{r,1}$',fontsize="15")
plt.text(0.492,0.55,r'$u_{r,2}$',fontsize="15")
plt.text(0.493,0.563,r'$u_{r,k}$',fontsize="15")





plt.axis('square')
plt.axis([0.48,0.65,0.485,0.58])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

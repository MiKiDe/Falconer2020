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
    
    return [(L[k]+v,L[k]) for k in range(m)],L

def E(J,s,m):
    J=[J]
    for q in range(s+1):
        A=[]
        for k in range(len(J)):
            u=J[k]
            plt.plot([u[0][0],u[1][0]],[u[0][1],u[1][1]])
            R=T(u,s,m)[0]
            for i in range(m):          #on construit la suite des nouveaux segments a chaque étape
                A.append(R[i])
            
        J=A
    return J
    
    
J=(np.array([1/2,-0.2]),np.array([1,-0.3]))
plt.plot([J[0][0],J[1][0]],[J[0][1],J[1][1]])
A,L=T(J,2,3)

for k in range(len(A)):
    w=A[k]
    plt.plot([w[0][0],w[1][0]],[w[0][1],w[1][1]])
    if k>0:
        c=L[k-1]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")
    else:
        c=J[0]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")       
  
plt.text(0.68,-0.26,r'$J$',fontsize="20")
plt.text(0.7,-0.16,r'$T( J )$',fontsize="20")
plt.text(0.51,-0.175,r'$\alpha$',fontsize="20")



plt.axis('square')
plt.axis([0.45,1.05,-0.32,-0.1])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

import numpy as np
import pylab as plt
import math

plt.style.use('seaborn-whitegrid') #Juste pour faire des figures un peu plus fancy qu'avec le style par défaut

""" Paramètres """
theta=np.pi/8
#beta = theta + np.pi/2 #J'ai mis beta plutot que theta pour pas confondre les angles quand on lira l'article (inutile pour le moment)
a=-1/np.tan(theta)

## Figure 1
u=np.array([1,a])/np.sqrt(1+a**2)
def p(y):
    return np.dot(u,y)*u

plt.figure(1)
plt.clf()

X=np.linspace(-1,1,2)
plt.plot(X,a*X,label=r'$l_{\theta}$',color='r')

x=np.array([0.6,0.4])
y=p(x)
plt.plot([x[0],y[0]],[x[1],y[1]])
plt.plot(x[0],x[1],marker='.',color='k')
plt.text(x[0]+0.02,x[1],'$x$',fontsize="15")
plt.plot(y[0],y[1],marker='+',color='k')
plt.text(y[0]+0.03,y[1]+0.01,r'$p_{\theta}(x)$',fontsize="15")
plt.text(0.13,0.67,r'$\theta$',fontsize="15")

plt.xticks([])
plt.yticks([])
plt.axis('square')
plt.axis([-1,1,-1,1])
plt.axhline(color='k')
plt.axvline(color='k')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()

## Figure 2
""" On code un segment par le couple des coordonnées de ses extremités """
plt.figure(2)
plt.clf()

def T(J,s,m):  #renvoie la liste des nouveaux segments
    alpha=np.pi/(4*s)
    a=J[1]
    b=J[0]
    if J[0][0]>J[1][0]: #On vérifie quelle extrémité est plus à gauche ou plus à droite, pour éviter les erreurs de saisie.
        a=J[0]
        b=J[1]
    l=np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)    #longueur de J
    L=[a+(k/m)*(b-a) for k in range(1,m+1)]     #liste des des origines des nouveaux segments
    beta=-np.arctan((a[1]-b[1])/(b[0]-a[0]))   #angle que fait J avec l'horizontale
    v=(l/(m*np.cos(alpha)))*np.array([np.cos(alpha+beta),np.sin(alpha+beta)])  #vecteur directeur des nouveaux segments
    return [(L[k]+v,L[k]) for k in range(m)],L


J=(np.array([1/2,-0.3]),np.array([1,-0.2]))
plt.plot([J[0][0],J[1][0]],[J[0][1],J[1][1]],'k')
A,L=T(J,2,3)

for k in range(len(A)):
    w=A[k]
    plt.plot([w[0][0],w[1][0]],[w[0][1],w[1][1]],color = 'purple')
    if k>0:
        c=L[k-1]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")
    else:
        c=J[1]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")

plt.text(0.70,-0.30,r'$J$',fontsize="20")
plt.text(0.7,-0.16,r'$T( J )$',fontsize="20",color = 'purple')
plt.text(0.6,-0.26,r'$\alpha$',fontsize="20")
plt.xticks([])
plt.yticks([])
plt.axis('square')
plt.axis([0.45,1.05,-0.32,-0.1])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

## Figure 3
plt.figure(3)
plt.clf()
plt.xticks([])
plt.yticks([])

a=1/np.tan(theta) #On redéfinit a ici parce qu'on veut un angle de signe opposé 
u=np.array([1,a])/np.sqrt(1+a**2)

J=(np.array([0.3,-0.3]),np.array([1,-0.2]))
plt.plot([J[0][0],J[1][0]],[J[0][1],J[1][1]],'k')
A,L=T(J,2,3)
for k in range(len(A)):
    w=A[k]
    plt.plot([w[0][0],w[1][0]],[w[0][1],w[1][1]],color = 'purple')
    if k>0:
        c=L[k-1]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")
    else:
        c=J[1]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")

X=np.linspace(-1,1,2)
plt.plot(X,a*X,label=r'$l_{\theta}$',color='r')

x,y=A[0][0],J[0]
z=p(x)
plt.plot([x[0],z[0]],[x[1],z[1]],'--',color ='grey')
z=p(y)
plt.plot([y[0],z[0]],[y[1],z[1]],'--',color = 'grey')


plt.text(0.68,-0.32,r'$J$',fontsize="15")
plt.text(0.6,-0.12,r'$T( J )$',fontsize="15",color = 'purple')
plt.text(-0.3,0.3,r'$-\frac{\pi}{4} \leq \theta \leq 0$',fontsize="15")
plt.text(0.8,-0.4,r'$0 \leq \alpha \leq \frac{\pi}{2}$',fontsize="15")
plt.text(-0.2,0,r'$l_{\theta}$',fontsize="18", color = 'r')

plt.axis('square')
plt.axis([-0.5,1.2,-0.5,0.5])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()
## Figure 4
a=1/np.tan(theta)
u=np.array([1,a])/np.sqrt(1+a**2)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.grid(False)
J=(np.array([1,-0.2]),np.array([0.2,-0.2]))
plt.plot([J[0][0],J[1][0]],[J[0][1],J[1][1]],'k')
A,L=T(J,2,2)
for k in range(len(A)):
    w=A[k]
    plt.plot([w[0][0],w[1][0]],[w[0][1],w[1][1]],color = 'purple')
    if k>0:
        c=L[k-1]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")
    else:
        c=J[0]
        plt.plot([w[0][0],c[0]],[w[0][1],c[1]],'--',color='k',linewidth="1")


X=np.linspace(-1,1,2)
plt.plot(X,a*X,label=r'$l_{\theta}$',color='r')

x,y=A[0][0],J[1]
z=p(x)
plt.plot([x[0],z[0]],[x[1],z[1]],'--',color ='grey')
z=p(y)
plt.plot([y[0],z[0]],[y[1],z[1]],'--',color = 'grey')
x=A[0][1]
z=p(x)
plt.plot([x[0],z[0]],[x[1],z[1]],'--',color ='grey')
x=A[1][0]
z=p(x)
plt.plot([x[0],z[0]],[x[1],z[1]],'--',color ='grey')
plt.axis('off')

plt.axis('square')
plt.axis([-0.5,1.2,-0.5,0.5])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()
## Figure 5
a=-1/np.tan(theta)
u=np.array([1,a])/np.sqrt(1+a**2)

fig = plt.figure(5)
plt.clf()
ax = fig.add_subplot(1, 1, 1)
ax.grid(False)
plt.xticks([])
plt.yticks([])
I=(np.array([1,1/2]),np.array([1/2,1/2]))

def E(J,s,m):
    J=[J]
    colors = ['slategrey','lightgray', 'darkgray', 'gray', 'dimgray', 'black']
    for q in range(s+1):
        A=[]
        for k in range(len(J)):
            u=J[k]
            plt.plot([u[0][0],u[1][0]],[u[0][1],u[1][1]],color = colors[q])
            R=T(u,s,m)[0]
            for i in range(m):          #on construit la suite des nouveaux segments a chaque étape
                A.append(R[i])

        J=A
    return J

E(I,5,2)

plt.text(0.73,0.48,r'$I$',fontsize="15")

plt.axis('square')
plt.axis([0.45,1.05,0.43,0.6])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()
plt.show()

## La vraie fonction E (pas encore finie, j'ai fais un algo en pseudo code mais ce sera fini plus tard)

def E(J,s,m):  #renvoie la liste des nouveaux segments
    alpha=np.pi/(4*s)
    a=J[0]
    b=J[1]
    beta=-np.arctan((a[1]-b[1])/(b[0]-a[0]))   #angle que fait J avec l'horizontale
    O = np.zeros((s,m)) #Tableau stockant la liste des origines où seront copiées les boûts de fractale
    P = [] #Tableau contenant
    l=np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)    #longueur de J
    for i in range(0,s):
        O[i]=[a+(k/m)*(b-a) for k in range(1,m+1)]     #liste des des origines des nouveaux segments
        v=(l/(m*np.cos(alpha))**i)*np.array([np.cos(i*alpha+beta),np.sin(i*alpha+beta)])  #vecteur directeur des nouveaux segments
    return [(L[k]+v,L[k]) for k in range(m)],L
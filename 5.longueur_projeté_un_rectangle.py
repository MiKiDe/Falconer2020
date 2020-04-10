import numpy as np
import pylab as plt

theta=np.pi/8
a=1/np.tan(theta)

u=np.array([1,a])/np.sqrt(1+a**2)
def p(z):
    return np.dot(u,z)*u

def relier(a,b,couleur='k'):
    plt.plot([a[0],b[0]],[a[1],b[1]],color=couleur)

X=np.linspace(0,1,200)
plt.plot(X,a*X,color='r')






x=np.array([0.4,0.4])
y=np.array([0.65,0.14])
relier(x,y)
diff=y-x
v=np.array([-diff[1],diff[0]])/2
relier(x,x+v)
relier(y,y+v)
relier(x+v,y+v)
relier(x+v,p(x+v),'g')
relier(x,p(x),'g')
relier(y,p(y),'g')

plt.text(0.42,0.9,r'$l_{\theta}$',fontsize="18")
plt.text(0.39,0.18,r'$r\alpha$',fontsize="15")
plt.text(0.67,0.41,r'$b$',fontsize="15")
plt.text(0.69,0.22,r'$h$',fontsize="15")
plt.text(0.09,0.57,r'$\theta$',fontsize="15")
plt.text(0.36,0.49,r'$d_1$',fontsize="15")
plt.text(0.31,0.34,r'$d_2$',fontsize="15")
plt.text(0.39,0.30,r'$\beta$',fontsize="15")





plt.axis('square')
plt.axis([0,1,0,1])
plt.axhline(color='k')
plt.axvline(color='k')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xticks([], [])
plt.yticks([], [])

plt.show()
import numpy as np
import pylab as plt

theta=np.pi/8
a=1/np.tan(theta)

u=np.array([1,a])/np.sqrt(1+a**2)
def p(y):
    return np.dot(u,y)*u


X=np.linspace(0,1,200)
plt.plot(X,a*X,label=r'$l_{\theta}$',color='r')
x=np.array([0.6,0.4])
y=p(x)
plt.plot([x[0],y[0]],[x[1],y[1]])
plt.plot(x[0],x[1],marker='+',color='k')
plt.text(x[0]+0.02,x[1],'$x$',fontsize="15")
plt.plot(y[0],y[1],marker='+',color='k')
plt.text(y[0]+0.03,y[1]+0.01,r'$p_{\theta}(x)$',fontsize="15")
plt.text(0.13,0.67,r'$\theta$',fontsize="15")



plt.axis('square')
plt.axis([0,1,0,1])
plt.axhline(color='k')
plt.axvline(color='k')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()
"""IPW estimate of per-window soft IoU plus original weighted binary loss.
The returned positive curvature is a surrogate, not an exact nonlocal Hessian.
"""
import numpy as np

def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-35,35)))

class WindowObjective:
    def __init__(self,y,w,windows,coefficient=.5):
        self.y=np.asarray(y,dtype=float);w=np.asarray(w,dtype=float)
        self.w=w/w.mean();self.coefficient=coefficient;self.n=len(y)
        self.bias=0.  # paired binary reference also uses boost_from_average=False
        self.group=np.zeros(len(y),dtype=int);self.inv=np.zeros(len(y));masses=[0.];P=[1.]
        for win in windows:
            a,b=win['start'],win['end']
            if not win['positive']:continue
            pos=self.y[a:b]>.5;assert pos.any()
            base=w[a:b][pos][0];assert np.allclose(w[a:b][pos],base)
            inv=w[a:b]/base;assert np.allclose(inv[pos],1)
            self.group[a:b]=len(masses);self.inv[a:b]=inv
            masses.append(w[a:b].sum());P.append(np.dot(inv,self.y[a:b]))
        self.mass=np.asarray(masses);self.P=np.asarray(P)
        assert np.isclose(self.mass.sum(),.8,atol=1e-5)
    def terms(self,z):
        p=sigmoid(np.asarray(z)+self.bias);g=self.group
        I=np.bincount(g,weights=self.inv*self.y*p,minlength=len(self.mass))
        D=self.P+np.bincount(g,weights=self.inv*(1-self.y)*p,minlength=len(self.mass))
        return p,I,D
    def loss(self,z):
        p,I,D=self.terms(z)
        return float(-np.dot(self.w,self.y*np.log(p)+(1-self.y)*np.log1p(-p))+self.coefficient*self.n*np.dot(self.mass,1-I/D))
    def gradient(self,z):
        p,I,D=self.terms(z);g=self.group
        dp=self.inv*((1-self.y)*I[g]/D[g]**2-self.y/D[g])
        scale=self.coefficient*self.n*self.mass[g]
        grad=self.w*(p-self.y)+scale*dp*p*(1-p)
        hess=self.w*p*(1-p)+scale*np.abs(dp)*p*(1-p)
        return grad,np.maximum(hess,1e-8)
    def __call__(self,z,dataset):
        assert len(z)==self.n and np.array_equal(dataset.get_label(),self.y)
        return self.gradient(z)

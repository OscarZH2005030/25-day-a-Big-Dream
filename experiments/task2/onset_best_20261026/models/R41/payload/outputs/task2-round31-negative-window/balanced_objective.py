"""Positive-window soft-IoU plus smooth upper-tail risk on sampled negatives.
This tail score is not the full-cell max or a calibrated event probability.
"""
import numpy as np
from window_objective import WindowObjective,sigmoid
class BalancedObjective(WindowObjective):
    def __init__(self,y,w,windows,coefficient=.5,negative_coefficient=.5,tau=.25):
        super().__init__(y,w,windows,coefficient)
        self.negative_coefficient=negative_coefficient;self.tau=tau
        idx=[];groups=[];shares=[];masses=[];w=np.asarray(w,dtype=float)
        for win in windows:
            if win['positive']:continue
            a,b=win['start'],win['end'];assert not self.y[a:b].any()
            mass=w[a:b].sum();idx.extend(range(a,b));groups.extend([len(masses)]*(b-a));shares.extend(w[a:b]/mass);masses.append(mass)
        self.nidx=np.asarray(idx,dtype=int);self.ngroup=np.asarray(groups,dtype=int);self.nshare=np.asarray(shares);self.nmass=np.asarray(masses)
        assert len(idx)>0 and np.isclose(self.nmass.sum(),.2,atol=1e-5)
    def negative_terms(self,z):
        v=(np.asarray(z)[self.nidx]+self.bias)/self.tau+np.log(self.nshare)
        mx=np.full(len(self.nmass),-np.inf);np.maximum.at(mx,self.ngroup,v)
        ex=np.exp(v-mx[self.ngroup]);total=np.bincount(self.ngroup,weights=ex,minlength=len(mx))
        score=self.tau*(mx+np.log(total));attention=ex/total[self.ngroup]
        return score,attention
    def loss(self,z):
        base=super().loss(z)
        if self.negative_coefficient==0:return base
        score,_=self.negative_terms(z)
        return float(base+self.negative_coefficient*self.n*np.dot(self.nmass,np.logaddexp(0,score)))
    def gradient(self,z):
        grad,hess=super().gradient(z)
        if self.negative_coefficient==0:return grad,hess
        score,a=self.negative_terms(z);p=sigmoid(score)[self.ngroup];scale=self.negative_coefficient*self.n*self.nmass[self.ngroup]
        grad[self.nidx]+=scale*p*a
        hess[self.nidx]+=scale*(p*(1-p)*a*a+p*a*(1-a)/self.tau)
        return grad,hess


import numpy as np
from scipy.special import expit
EPS=1e-6
def baseline_logit(peak):
    p=np.clip(np.asarray(peak,dtype='float64'),EPS,1-EPS);return np.log(p)-np.log1p(-p)
def residual_derivatives(delta,offset,y,w):
    p=expit(offset+delta);return w*(p-y),w*p*(1-p)
def residual_loss(delta,offset,y,w):
    eta=offset+delta;return float(np.dot(w,np.logaddexp(0,eta)-y*eta))
def corrected_peak(peak,delta):
    peak=np.asarray(peak,dtype='float32');delta=np.asarray(delta,dtype='float64');out=peak.copy();use=(peak>EPS)&(delta!=0)
    out[use]=expit(baseline_logit(peak[use])+delta[use]).astype('float32');assert np.isfinite(out).all() and (out>=0).all() and (out<=1).all();return out

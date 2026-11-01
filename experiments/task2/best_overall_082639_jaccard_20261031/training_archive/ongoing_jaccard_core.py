"""Window Soft-IoU loss with exact gradient and positive curvature surrogate."""
import numpy as np

def sigmoid(z):return np.exp(-np.logaddexp(0.,-np.asarray(z,dtype='float64')))

def loss_gradient(z,y,w,group,strength):
    y=np.asarray(y,dtype='float64');w=np.asarray(w,dtype='float64');group=np.asarray(group,dtype='int32');p=sigmoid(z);n=int(group.max())+1
    mass=np.bincount(group,weights=w,minlength=n);inter=np.bincount(group,weights=w*p*y,minlength=n);union=np.bincount(group,weights=w*(y+p*(1-y)),minlength=n)+1e-9
    j=inter/union;scale=mass[group]/union[group];pq=p*(1-p)
    loss=float(np.sum(w*(np.logaddexp(0.,z)-y*z))+strength*np.sum(mass*(1-j)))
    gradient=w*(p-y)+strength*scale*w*pq*(j[group]*(1-y)-y)
    # Soft-IoU's exact Hessian can be negative; use a positive Fisher-like proxy.
    hessian=w*pq+strength*scale*w*pq*(y+j[group]*(1-y))
    return loss,gradient,np.maximum(hessian,1e-8)

def objective(w,group,strength):
    def f(z,data):
        _,g,h=loss_gradient(z,data.get_label(),w,group,strength)
        return g,h
    return f

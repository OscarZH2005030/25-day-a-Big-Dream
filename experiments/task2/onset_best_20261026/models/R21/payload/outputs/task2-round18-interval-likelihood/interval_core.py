"""A distribution over empty/one-contiguous queue intervals, with partial labels."""
import numpy as np
PATTERNS=np.zeros((22,6),dtype='float64')
INTERVALS=[None]
for start in range(6):
    for end in range(start,6):
        PATTERNS[len(INTERVALS),start:end+1]=1.;INTERVALS.append([start+1,end+1])
BITS=(np.uint32(1)<<np.arange(22,dtype='uint32'))
ALL=int(BITS.sum())
def encode(queue,mask):
    matches=((PATTERNS[:,None,:]==queue.T[None])|~mask.T[None]).all(2)
    return (matches.T*BITS).sum(1).astype('uint32')
def allowed(codes):return (np.asarray(codes,dtype='uint32')[:,None]&BITS[None])!=0
def softmax(raw):
    v=np.asarray(raw,dtype='float64');v=v-v.max(1,keepdims=True);v=np.exp(v);return v/v.sum(1,keepdims=True)
def logsumexp(raw):
    top=raw.max(1);return top+np.log(np.exp(raw-top[:,None]).sum(1))
def loss_grad(raw,mask,weight=None):
    masked=np.where(mask,raw,-np.inf);p=softmax(raw);posterior=softmax(masked);g=p-posterior
    # Hessian = Cov(p)-Cov(p | allowed). 2*diag(p*(1-p)) is a positive
    # diagonal upper approximation; it is deliberately NOT exact Newton.
    h=np.maximum(2*p*(1-p),1e-4);loss=logsumexp(raw)-logsumexp(masked)
    if weight is not None:g*=weight[:,None];h*=weight[:,None];value=float(np.dot(weight,loss)/weight.sum())
    else:value=float(loss.mean())
    return value,g,h
class PartialObjective:
    def __init__(self):self.mask=None;self.weights=None;self.latest_loss=None
    def __call__(self,raw,dataset):
        if self.mask is None:
            codes=dataset.get_label().astype('uint32');assert (codes>0).all() and (codes<ALL).all();self.mask=allowed(codes);weights=dataset.get_weight();self.weights=np.ones(len(codes)) if weights is None else weights.astype('float64')
        assert raw.shape==(len(self.mask),22)
        self.latest_loss,g,h=loss_grad(raw,self.mask,self.weights);assert np.isfinite(self.latest_loss) and np.isfinite(g).all() and np.isfinite(h).all()
        return g.astype('float32'),h.astype('float32')
def queue_probability(model,x):
    raw=model.predict(x,raw_score=True,num_threads=4);assert raw.shape==(len(x),22);p=softmax(raw)
    return (p@PATTERNS).astype('float32').T

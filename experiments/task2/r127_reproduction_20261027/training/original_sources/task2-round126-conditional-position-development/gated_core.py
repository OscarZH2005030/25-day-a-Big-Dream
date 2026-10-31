"""Fixed majority-mode concentration gate; not calibrated predictive certainty."""
import numpy as np
from timing_core import adjust,unseen_queue
def gated_adjust(s,base,first_prob,alpha):
    if s['condition']!=0 or alpha==0:return base.copy()
    proposed=adjust(s,base,first_prob,alpha)
    use=(np.max(first_prob,axis=1)>=.5)&unseen_queue(s)
    out=base.copy();out[:,use]=proposed[:,use]
    assert np.array_equal(out.max(0),base.max(0)) and np.array_equal(out[:,~use],base[:,~use])
    return out

def concentrated_adjust(s,parent,anchor,first_prob,alpha):
    if s['condition']!=0 or alpha==0:return parent.copy()
    assert np.array_equal(parent.max(0),anchor.max(0))
    proposed=adjust(s,anchor,first_prob,alpha)
    use=(np.max(first_prob,axis=1)>=.5)&unseen_queue(s)
    out=parent.copy();out[:,use]=proposed[:,use]
    assert np.array_equal(out.max(0),parent.max(0)) and np.array_equal(out[:,~use],parent[:,~use])
    return out

import unittest
import numpy as np
from ongoing_jaccard_core import loss_gradient,sigmoid

class JaccardChecks(unittest.TestCase):
    def test_exact_gradient_against_scalar_finite_differences(self):
        z=np.array([-.8,.3,1.7,-1.2,.7]);y=np.array([1,0,1,0,0]);w=np.array([.4,1.8,.7,2.,.5]);group=np.array([0,0,1,1,1]);_,g,h=loss_gradient(z,y,w,group,.75);numeric=[]
        for i in range(len(z)):
            a=z.copy();b=z.copy();a[i]+=1e-5;b[i]-=1e-5;numeric.append((loss_gradient(a,y,w,group,.75)[0]-loss_gradient(b,y,w,group,.75)[0])/2e-5)
        np.testing.assert_allclose(g,numeric,atol=1e-8,rtol=1e-7);self.assertTrue((h>0).all())
    def test_empty_windows_keep_binary_negative_supervision(self):
        z=np.array([-3.,1.]);y=np.zeros(2);w=np.array([1.,2.]);group=np.zeros(2,dtype=int);_,g,_=loss_gradient(z,y,w,group,.75);np.testing.assert_allclose(g,w*sigmoid(z))
    def test_zero_strength_is_weighted_logistic(self):
        z=np.array([-4.,2.,0.]);y=np.array([1,0,1]);w=np.array([2.,.3,1.]);_,g,h=loss_gradient(z,y,w,np.array([0,0,1]),0);p=sigmoid(z);np.testing.assert_allclose(g,w*(p-y));np.testing.assert_allclose(h,w*p*(1-p))
    def test_extreme_logits_finite(self):
        loss,g,h=loss_gradient(np.array([-1000.,1000.]),np.array([1,0]),np.ones(2),np.array([0,0]),.75);self.assertTrue(np.isfinite(loss));self.assertTrue(np.isfinite(g).all());self.assertTrue((h>0).all())

if __name__=='__main__':unittest.main()

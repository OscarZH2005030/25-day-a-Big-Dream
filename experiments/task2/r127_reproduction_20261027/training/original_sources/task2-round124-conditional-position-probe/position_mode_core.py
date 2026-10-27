import numpy as np
from coverage_data import matrix
K=8
RIDGE=1e-3

def canonical(q):
    q=q.copy()
    for j in range(q.shape[1]):
        if q[np.argmax(abs(q[:,j])),j]<0:q[:,j]*=-1
    return q

def extras(s,basis,mode):
    assert mode in ['position','random'];phi=basis[mode];x=s['x'];n=x.shape[1];assert phi.shape==(n,K);support=basis['supported'];parts=[];fractions=[]
    for t in range(13):
        for ch,missing in [(0,4),(1,5),(2,6)]:
            seen=(x[t,:,missing]<.5)&(x[t,:,3]>=.75)&(x[t,:,7]>.5)&support;v=x[t,:,ch];fractions.append(float(seen.sum()/max(int(support.sum()),1)))
            if seen.any():
                F=phi[seen];coef=np.linalg.solve(F.T@F+RIDGE*np.eye(K),F.T@v[seen])
            else:coef=np.zeros(K)
            parts.append(phi*coef[None,:])
    result=np.column_stack(parts+[np.tile(fractions,(n,1)),phi]).astype('float32');assert result.shape==(n,359) and np.isfinite(result).all();return result

def feature_matrix(s,basis,mode):
    x=matrix(s);g=extras(s,basis,mode);out=np.column_stack([x,np.tile(g,(6,1))]).astype('float32');assert out.shape==(len(x),914) and np.array_equal(out[:,:555],x);return out

def make_basis(order,known,seed):
    n=len(order);ids=np.flatnonzero(known);unique=len(np.unique(order[ids]));B=np.zeros((n,K));R=np.zeros_like(B);condition=float('inf');support=np.asarray(known,dtype=bool)
    if len(ids)>=16 and unique>=16:
        z=2*(order[ids]-order[ids].min())/(order[ids].max()-order[ids].min())-1;V=np.polynomial.legendre.legvander(z,K-1);condition=float(np.linalg.cond(V));B[ids]=canonical(np.linalg.qr(V)[0]);rng=np.random.default_rng(seed);const=np.ones(len(ids))/np.sqrt(len(ids));r=rng.normal(size=(len(ids),K-1));r-=r.mean(0);R[ids]=canonical(np.linalg.qr(np.column_stack([const,r]))[0]);assert np.allclose(B.T@B,np.eye(K),atol=1e-10) and np.allclose(R.T@R,np.eye(K),atol=1e-10)
    gate=bool(n>=16 and len(ids)/n>=.8 and unique>=16 and condition<=1e8)
    return dict(position=B,random=R,supported=support),dict(nodes=n,known_order_nodes=len(ids),known_fraction=len(ids)/n,distinct_public_positions=unique,condition_number=condition,training_gate=gate,geometry='fixed degree0..7 Legendre position basis of EXPLICIT PUBLIC order_index, empirical orthonormalization; no invented edge, adjacency, physical distance, or travel-time claim')

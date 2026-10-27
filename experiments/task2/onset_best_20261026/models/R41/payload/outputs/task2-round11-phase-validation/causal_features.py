"""186特征思想的历史因果重建；不导入Task1运行时或任何未来数组。"""
import numpy as np
from task2q.baselines import tree_features

def enhanced(s, net):
    x=s['x']; n=x.shape[1]; cols=[]
    # 过去多尺度稳健统计、可靠性、斜率；所有切片截止T。
    for ch,missing in [(0,4),(1,5),(2,6)]:
        a=x[:,:,ch]; valid=x[:,:,missing]<.5
        carry=np.zeros_like(a); last=np.full(n,1. if ch==0 else 0.)
        for t in range(13):
            last=np.where(valid[t],a[t],last);carry[t]=last
        for w in [3,6,13]:
            z=carry[-w:]; q=np.quantile(z,[.25,.5,.75],axis=0)
            cols.extend([z.mean(0),z.std(0),z.min(0),z.max(0),q[1],q[2]-q[0],np.median(abs(z-q[1]),axis=0),valid[-w:].mean(0),(z[-1]-z[0])/max(w-1,1)])
        cols.extend([carry[-1]-carry[-4],carry[-4]-carry[-7],carry[-1]-2*carry[-4]+carry[-7]])
    speed=x[:,:,0];valid=x[:,:,4]<.5;cut=net['threshold']/net['free_speed']
    queued=(speed<=cut)&valid
    run=np.zeros(n)
    for q in queued:run=np.where(q,run+1,0)
    cols.extend([run/13,queued.sum(0)/13,queued[-3:].sum(0)/3,queued[-6:].sum(0)/6])
    # 有向上下游1/2/3跳传播，禁止按link_id顺序虚构邻居。
    edges=net['edges']; latest=np.ones(n)
    for t in range(13):latest=np.where(valid[t],speed[t],latest)
    pressure=np.maximum(cut-latest,0)
    for reverse in [False,True]:
        a,b=edges if reverse else edges[::-1]
        z=np.column_stack([latest,pressure,queued[-3:].mean(0),x[-1,:,1],x[-1,:,2]])
        for hop in range(3):
            sums=np.zeros_like(z);counts=np.zeros(n);maximum=np.zeros_like(z)
            np.add.at(sums,b,z[a]);np.add.at(counts,b,1);np.maximum.at(maximum,b,z[a])
            z=np.divide(sums,counts[:,None],out=np.zeros_like(sums),where=counts[:,None]>0)
            cols.extend([*z.T,*maximum.T,counts])
    # 走廊拥堵面积/发展速度及位置，均来自可见历史。
    for t in [-1,-4,-7,-13]:
        cols.extend([np.full(n,queued[t].mean()),np.full(n,speed[t].mean())])
    stamp=__import__('pandas').Timestamp(s['origin'])
    cols.extend([np.full(n,stamp.dayofweek/6),np.arange(n)/max(n-1,1)])
    base=np.column_stack(cols).astype('float32')
    return np.concatenate([tree_features(s,net),np.tile(base,(6,1))],axis=1)

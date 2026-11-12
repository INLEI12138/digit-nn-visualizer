import numpy as np
d=np.load('data2.npz');X,Y=d['X'],d['Y'];Xe,Ye=d['Xe'],d['Ye']
rng=np.random.default_rng(1);p=rng.permutation(len(X));X,Y=X[p],Y[p]
Xt,Yt,Xv,Yv=X[:-3000],Y[:-3000],X[-3000:],Y[-3000:]
sizes=[784,64,32,10]
W=[rng.normal(0,np.sqrt(2/a),(a,b)).astype(np.float32) for a,b in zip(sizes,sizes[1:])]
B=[np.zeros(b,np.float32) for b in sizes[1:]]
P=W+B;m=[np.zeros_like(q) for q in P];v=[np.zeros_like(q) for q in P];t=0
def fwd(x,drop=False):
    hs=[x]
    for i in range(3):
        z=hs[-1]@W[i]+B[i]
        if i<2:
            z=np.maximum(z,0)
            if drop: z=z*(rng.random(z.shape)>0.15)/0.85
        hs.append(z)
    return hs
for ep in range(25):
    lr=2e-3*(0.9**ep);idx=rng.permutation(len(Xt))
    for s in range(0,len(idx),128):
        b=idx[s:s+128];x=Xt[b]
        x=np.clip(x+rng.normal(0,0.05,x.shape)*(rng.random()<.3),0,1)
        hs=fwd(x,True);z=hs[-1];z=z-z.max(1,keepdims=True);pr=np.exp(z);pr/=pr.sum(1,keepdims=True)
        g=pr;g[np.arange(len(b)),Yt[b]]-=1;g/=len(b)
        gW=[None]*3;gB=[None]*3
        for i in (2,1,0):
            gW[i]=hs[i].T@g+1e-4*W[i];gB[i]=g.sum(0)
            if i>0: g=(g@W[i].T)*(hs[i]>0)
        t+=1
        for k,(q,gq) in enumerate(zip(P,gW+gB)):
            m[k]=0.9*m[k]+0.1*gq;v[k]=0.999*v[k]+0.001*gq*gq
            q-=lr*(m[k]/(1-0.9**t))/(np.sqrt(v[k]/(1-0.999**t))+1e-8)
    acc=(fwd(Xv)[-1].argmax(1)==Yv).mean()
    print(ep,round(acc,4),'real',round((fwd(Xe)[-1].argmax(1)==Ye).mean(),4),flush=True)
np.savez('weights.npz',*W,*B)

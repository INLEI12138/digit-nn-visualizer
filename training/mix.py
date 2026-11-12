import numpy as np,sys
sys.path.insert(0,'.')
import gen
from gen import preprocess,augment
from sklearn.datasets import load_digits
from scipy import ndimage
d=load_digits();rng=np.random.default_rng(5);p=rng.permutation(len(d.images))
tr,te=p[:900],p[900:]
def up(img): 
    a=ndimage.zoom(img/16,14,order=3);a=np.clip(a,0,1)
    return np.pad(a,(S:=0,)) if False else a
Xa=[];Ya=[]
for k in range(12):
    for i in tr:
        a=up(d.images[i]);a=np.pad(a,(112-a.shape[0])//2)
        a=augment(a) if k else a
        a=np.clip((a-0.25)*1.6,0,1) if rng.random()<.6 else a  # sharpen
        Xa.append(preprocess(a).ravel());Ya.append(d.target[i])
Xe=[preprocess(np.pad(up(d.images[i]),(112-112)//2)).ravel() for i in te]
D=np.load('data.npz')
np.savez_compressed('data2.npz',X=np.vstack([D['X'],np.array(Xa,np.float32)]),Y=np.concatenate([D['Y'],Ya]),Xe=np.array(Xe,np.float32),Ye=d.target[te])
print(len(Xa))

import numpy as np, glob, random, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage
rng=np.random.default_rng(0); random.seed(0)
S=112
def catmull(pts,n=12):
    pts=[pts[0]]+pts+[pts[-1]];out=[]
    for i in range(1,len(pts)-2):
        p0,p1,p2,p3=[np.array(p) for p in pts[i-1:i+3]]
        for t in np.linspace(0,1,n,endpoint=False):
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3))
    out.append(np.array(pts[-2]));return out
def arc(cx,cy,rx,ry,a0,a1,n=10):
    return [(cx+rx*math.cos(math.radians(a)),cy+ry*math.sin(math.radians(a))) for a in np.linspace(a0,a1,n)]
def J(p,s=0.04): return [(x+rng.normal(0,s),y+rng.normal(0,s)) for x,y in p]
def strokes(d):
    v=rng.integers(0,3)
    if d==0:
        rx=rng.uniform(.25,.4); a0=rng.uniform(-110,-60)
        return [J(arc(.5,.5,rx,.45,a0,a0+360+rng.uniform(-10,25),12),.03)]
    if d==1:
        if v==0: return [J([(.5,.05),(.5,.95)],.05)]
        if v==1: return [J([(.32,.25),(.5,.05),(.5,.95)],.04)]
        return [J([(.35,.22),(.5,.05),(.5,.95)],.04),J([(.3,.95),(.7,.95)],.03)]
    if d==2:
        s=[(.18,.3),(.35,.07),(.62,.05),(.8,.25),(.7,.5),(.4,.75),(.15,.95)]
        if v==2: s=s[:-1]+[(.2,.9),(.35,.85),(.15,.95)]  # loop-ish ignored
        return [J(s,.035)+J([(.18,.95),(.85,.95)],.03)[1:] if False else J(s,.035), J([(.15,.95),(.85,.95)],.03)]
    if d==3:
        if v==0: return [J([(.2,.15),(.5,.03),(.78,.2),(.72,.38),(.45,.48)],.03), J([(.45,.48),(.78,.6),(.82,.8),(.55,.97),(.18,.85)],.03)]
        return [J([(.2,.15),(.5,.03),(.78,.2),(.72,.38),(.45,.48),(.78,.6),(.82,.8),(.55,.97),(.18,.85)],.03)]
    if d==4:
        if v==0: return [J([(.62,.95),(.62,.05),(.12,.68),(.88,.68)],.035)]
        return [J([(.25,.05),(.18,.6),(.85,.6)],.04),J([(.65,.25),(.65,.95)],.04)]
    if d==5:
        return [J([(.78,.06),(.28,.06),(.22,.45)],.03), J([(.22,.45),(.5,.36),(.78,.5),(.8,.78),(.55,.96),(.18,.86)],.03)]
    if d==6:
        cx,cy,r=.5,.7,rng.uniform(.22,.3)
        return [J([(.72,.05),(.45,.15),(.26,.45)],.03)+J(arc(cx,cy,r,.26,180,540,12),.02)]
    if d==7:
        s=[J([(.15,.06),(.85,.06),(.6,.45),(.42,.95)],.04)]
        if v==2: s.append(J([(.35,.5),(.8,.5)],.03))
        return s
    if d==8:
        if v==0: return [J(arc(.5,.27,.22,.22,90,450,10),.025),J(arc(.5,.72,.27,.24,-90,270,10),.025)]
        p=[(.5,.5),(.75,.3),(.5,.05),(.25,.25),(.5,.5),(.78,.72),(.5,.96),(.22,.72),(.5,.5)]
        return [J(p,.03)]
    if d==9:
        r=rng.uniform(.2,.28)
        c=J(arc(.5,.3,r,.25,0,360,11),.025)
        tail=J([(.5+r,.3),(.5+r,.6),(.6+r*0.3,.95)] if v else [(.5+r,.3),(.68,.95)],.03)
        return [c,tail]
def render_strokes(d):
    im=Image.new('L',(S,S),0);dr=ImageDraw.Draw(im)
    w=int(rng.uniform(5,14))
    for st in strokes(d):
        pts=catmull([tuple(p) for p in st]) if len(st)>2 else [np.array(p) for p in st]
        P=[(float(x)*80+16,float(y)*80+16) for x,y in pts]
        dr.line(P,fill=255,width=w,joint='curve')
        for x,y in (P[0],P[-1]): dr.ellipse([x-w/2,y-w/2,x+w/2,y+w/2],fill=255)
    return np.array(im,np.float32)/255
fonts=[f for f in glob.glob('/usr/share/fonts/**/*.[ot]tf',recursive=True)
       if any(k in f for k in ['DejaVuSans','Liberation','Free','Inter-','Poppins','Carlito','Caladea','Lora','Loma','DejaVuSerif'])
       and 'Mono' not in f and 'Black' not in f and 'Thin' not in f]
fonts=[ImageFont.truetype(f,72) for f in fonts]
print(len(fonts),'fonts')
def render_font(d):
    im=Image.new('L',(S,S),0);dr=ImageDraw.Draw(im)
    dr.text((S/2,S/2),str(d),font=random.choice(fonts),fill=255,anchor='mm')
    a=np.array(im,np.float32)/255
    k=rng.integers(-1,4)
    if k>0: a=ndimage.grey_dilation(a,size=(k*2+1,k*2+1))
    return a
def augment(a):
    ang=rng.uniform(-14,14);sh=rng.uniform(-.3,.3);sx=rng.uniform(.75,1.25)
    th=math.radians(ang)
    M=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])@np.array([[1,sh],[0,1]])@np.diag([1,sx])
    c=np.array([S/2,S/2]); off=c-M@c
    a=ndimage.affine_transform(a,M,offset=off,order=1)
    if rng.random()<.5:  # elastic
        dx=ndimage.gaussian_filter(rng.uniform(-1,1,(S,S)),8)*60;dy=ndimage.gaussian_filter(rng.uniform(-1,1,(S,S)),8)*60
        y,x=np.meshgrid(np.arange(S),np.arange(S),indexing='ij')
        a=ndimage.map_coordinates(a,[y+dy,x+dx],order=1)
    return a
def preprocess(a):
    # identical to JS: bbox -> fit 20x20 keep aspect -> paste into 28x28 centered by center of mass
    ys,xs=np.where(a>0.1)
    if len(ys)==0: return np.zeros((28,28),np.float32)
    a=a[ys.min():ys.max()+1,xs.min():xs.max()+1]
    h,w=a.shape;s=20/max(h,w);nh,nw=max(1,round(h*s)),max(1,round(w*s))
    im=Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize((nw,nh),Image.BILINEAR)
    b=np.array(im,np.float32)/255
    out=np.zeros((28,28),np.float32)
    cy,cx=ndimage.center_of_mass(b)
    oy=int(round(14-cy));ox=int(round(14-cx))
    oy=min(max(oy,0),28-nh);ox=min(max(ox,0),28-nw)
    out[oy:oy+nh,ox:ox+nw]=b
    return out
if __name__=='__main__':
    N=int(__import__('sys').argv[1]);X=np.zeros((N,784),np.float32);Y=np.zeros(N,np.int64)
    for i in range(N):
        d=i%10
        a=render_strokes(d) if rng.random()<0.65 else render_font(d)
        X[i]=preprocess(augment(a)).ravel();Y[i]=d
    np.savez_compressed('data.npz',X=X,Y=Y)
    # preview
    g=np.zeros((28*10,28*20))
    for r in range(10):
        idx=np.where(Y==r)[0][:20]
        for c,j in enumerate(idx): g[r*28:(r+1)*28,c*28:(c+1)*28]=X[j].reshape(28,28)
    Image.fromarray((g*255).astype(np.uint8)).resize((560*2,280*2),Image.NEAREST).save('preview.png')

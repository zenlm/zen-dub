#!/usr/bin/env python3
"""Generate visual lip-sync inspection artifacts: contact sheet, mouth-zoom strip, side-by-side."""
import sys, os, glob, json, argparse
import numpy as np, cv2

def load_frames(d):
    return [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d,"*.png")))]

def label(img, text, scale=0.6, color=(255,255,255)):
    img=img.copy()
    cv2.rectangle(img,(0,0),(img.shape[1], int(26*scale/0.6)),(0,0,0),-1)
    cv2.putText(img,text,(4,int(18*scale/0.6)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    return img

def contact_sheet(frames, cols=8, rows=4, tile_w=180, title="contact"):
    n=cols*rows
    idx=np.linspace(0,len(frames)-1,n).astype(int)
    h,w=frames[0].shape[:2]; th=int(tile_w*h/w)
    sheet=np.zeros((rows*th, cols*tile_w,3),np.uint8)
    for k,fi in enumerate(idx):
        r,c=divmod(k,cols)
        t=cv2.resize(frames[fi],(tile_w,th))
        t=label(t,f"f{fi} {fi/25:.1f}s",0.45)
        sheet[r*th:(r+1)*th, c*tile_w:(c+1)*tile_w]=t
    return sheet

def mouth_strip(frames, cx, cy, cw, ch, count=10, zoom=2):
    idx=np.linspace(0,len(frames)-1,count).astype(int)
    x1=max(0,cx-cw//2); y1=max(0,cy-ch//2)
    crops=[]
    for fi in idx:
        crop=frames[fi][y1:y1+ch, x1:x1+cw]
        crop=cv2.resize(crop,(cw*zoom,ch*zoom),interpolation=cv2.INTER_NEAREST)
        crop=label(crop,f"f{fi}",0.5)
        crops.append(crop)
    return np.hstack(crops)

def sidebyside_montage(src_frames, dub_frames, align_fn, count=6, tile_w=240):
    idx=np.linspace(0,len(dub_frames)-1,count).astype(int)
    h,w=dub_frames[0].shape[:2]; th=int(tile_w*h/w)
    rows=[]
    for fi in idx:
        si=align_fn(fi)
        s=cv2.resize(src_frames[si],(tile_w,th)); s=label(s,f"SRC f{si}",0.5,(180,255,180))
        d=cv2.resize(dub_frames[fi],(tile_w,th)); d=label(d,f"DUB f{fi}",0.5,(180,220,255))
        sep=np.full((th,3,3),(0,255,255),np.uint8)
        rows.append(np.hstack([s,sep,d]))
    # stack 2 per row
    out=[]
    for i in range(0,len(rows),2):
        pair=rows[i:i+2]
        if len(pair)==2:
            vsep=np.full((4,pair[0].shape[1],3),(0,255,255),np.uint8)
            out.append(np.vstack([pair[0],vsep,pair[1]]))
        else:
            out.append(pair[0])
    maxw=max(r.shape[1] for r in out)
    out=[cv2.copyMakeBorder(r,0,0,0,maxw-r.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0)) for r in out]
    return np.vstack(out)

if __name__=="__main__":
    WORK=os.environ.get("WORK","/home/z/work/sw-perf/ml/lipsync-eval/work")
    OUT=os.environ.get("OUT", sys.argv[1] if len(sys.argv)>1 else WORK)
    PREFIX=os.environ.get("PREFIX","zen-dub")
    DUB=os.environ.get("DUB_FRAMES", f"{WORK}/frames_fullnative")
    SRC=os.environ.get("SRC_FRAMES", f"{WORK}/frames_source")
    BBOX=os.environ.get("BBOX", f"{WORK}/signal_bbox.json")
    bbox=json.load(open(BBOX))
    cx,cy=bbox["center"]
    dub=load_frames(DUB)
    src=load_frames(SRC) if os.path.isdir(SRC) and glob.glob(os.path.join(SRC,"*.png")) else []
    print(f"dub={len(dub)} src={len(src)} mouth center=({cx},{cy})")

    # 1. contact sheet
    cs=contact_sheet(dub,8,4,180)
    cv2.imwrite(f"{OUT}/{PREFIX}-contact.png",cs); print("wrote contact", cs.shape)

    if not src:
        print("no source frames -> skipping side-by-side montage"); sys.exit(0)
    # 3. side-by-side montage: need alignment (boomerang loop handled by nearest-upper-face)
    h,w=src[0].shape[:2]
    su=[s[:int(h*0.45)].astype(np.float32) for s in src]
    cache={}
    def align(fi):
        if fi in cache: return cache[fi]
        u=dub[fi][:int(h*0.45)].astype(np.float32);b=0;bv=1e18
        for i,s in enumerate(su):
            v=np.mean((u-s)**2)
            if v<bv:bv=v;b=i
        cache[fi]=b; return b
    sb=sidebyside_montage(src,dub,align,count=6,tile_w=240)
    cv2.imwrite(f"{OUT}/{PREFIX}-sidebyside.png",sb); print("wrote sidebyside montage", sb.shape)

#!/usr/bin/env python3
"""Recover the lip-sync write region (mouth bbox + center) empirically by diffing a dubbed video's
frames against its source frames on loop-aligned pairs. The MuseTalk pipeline only modifies the
lower-face/mouth paste region, so the accumulated abs-diff localizes exactly where it wrote.
Writes <work>/signal_bbox.json and <work>/diff_heatmap.png.
If no source is available, falls back to a lower-center prior (frac 0.48,0.375).
"""
import sys, os, glob, json
import numpy as np, cv2

def recover(dub_dir, src_dir, out_dir):
    dub=sorted(glob.glob(os.path.join(dub_dir,"*.png")))
    d0=cv2.imread(dub[0]); h,w=d0.shape[:2]
    bbox={"w":w,"h":h}
    if src_dir and glob.glob(os.path.join(src_dir,"*.png")):
        src=[cv2.imread(p) for p in sorted(glob.glob(os.path.join(src_dir,"*.png")))]
        su=[s[:int(h*0.45)].astype(np.float32) for s in src]
        def near(img):
            u=img[:int(h*0.45)].astype(np.float32);b=0;bv=1e18
            for i,s in enumerate(su):
                v=np.mean((u-s)**2)
                if v<bv:bv=v;b=i
            return b,bv
        acc=np.zeros((h,w)); n=0
        for di in range(0,len(dub),5):
            d=cv2.imread(dub[di]); si,bv=near(d)
            if bv>40: continue
            acc+=np.abs(d.astype(np.float32)-src[si].astype(np.float32)).mean(2); n+=1
        if n>0:
            acc/=n
            floor=np.median(acc); sig=np.clip(acc-floor*1.5,0,None)
            mask=(sig>sig.max()*0.08).astype(np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
            num,lab,stats,cent=cv2.connectedComponentsWithStats(mask)
            if num>1:
                big=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
                x,y,ww,hh,_=stats[big]
                bbox["signal_box"]=[int(x),int(y),int(x+ww),int(y+hh)]
                bbox["center"]=[int(x+ww//2),int(y+hh//2)]
                bbox["method"]="source-diff"
            hm=cv2.applyColorMap((acc/acc.max()*255).astype(np.uint8),cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(out_dir,"diff_heatmap.png"),hm)
    if "center" not in bbox:  # fallback prior
        cx,cy=int(0.48*w),int(0.375*h)
        bbox["center"]=[cx,cy]; bbox["signal_box"]=[cx-30,cy-25,cx+30,cy+25]
        bbox["method"]="prior (no source / no signal)"
    json.dump(bbox, open(os.path.join(out_dir,"signal_bbox.json"),"w"))
    return bbox

if __name__=="__main__":
    dub_dir=sys.argv[1]; src_dir=sys.argv[2] if len(sys.argv)>2 and sys.argv[2]!="-" else None
    out_dir=sys.argv[3] if len(sys.argv)>3 else os.path.dirname(dub_dir)
    print(json.dumps(recover(dub_dir,src_dir,out_dir),indent=2))

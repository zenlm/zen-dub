#!/usr/bin/env python3
"""Tight mouth-zoom strip + seam-overlay row, for viseme + blend-seam inspection."""
import sys, glob, json, os
import numpy as np, cv2
WORK=os.environ.get("WORK","/home/z/work/sw-perf/ml/lipsync-eval/work")
OUT=os.environ.get("OUT", sys.argv[1] if len(sys.argv)>1 else WORK)
PREFIX=os.environ.get("PREFIX","zen-dub")
BBOX=os.environ.get("BBOX", f"{WORK}/signal_bbox.json")
DUB=os.environ.get("DUB_FRAMES", f"{WORK}/frames_fullnative")
bbox=json.load(open(BBOX))
cx,cy=bbox["center"]; sb=bbox["signal_box"]
dub=sorted(glob.glob(f"{DUB}/*.png"))
frames=[cv2.imread(p) for p in dub]
h,w=frames[0].shape[:2]

# Tight mouth crop: 150w x 130h centered, captures upper-lip->chin & full mouth.
CW,CH,ZOOM,N=150,130,3,12
idx=np.linspace(0,len(frames)-1,N).astype(int)
x1=max(0,cx-CW//2); y1=max(0,cy-CH//2+8)  # nudge down to center lips
def lab(img,t):
    cv2.rectangle(img,(0,0),(img.shape[1],18),(0,0,0),-1)
    cv2.putText(img,t,(3,13),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,255),1,cv2.LINE_AA);return img
row1=[]  # plain zoom
row2=[]  # with paste-seam box overlay
for fi in idx:
    c=frames[fi][y1:y1+CH, x1:x1+CW]
    z=cv2.resize(c,(CW*ZOOM,CH*ZOOM),interpolation=cv2.INTER_CUBIC)
    row1.append(lab(z.copy(),f"f{fi} {fi/25:.1f}s"))
    # overlay signal box (paste core) coords in crop space
    z2=z.copy()
    bx1=(sb[0]-x1)*ZOOM; by1=(sb[1]-y1)*ZOOM; bx2=(sb[2]-x1)*ZOOM; by2=(sb[3]-y1)*ZOOM
    cv2.rectangle(z2,(bx1,by1),(bx2,by2),(0,0,255),1)
    row2.append(lab(z2,f"f{fi}"))
strip=np.vstack([np.hstack(row1),np.hstack(row2)])
cv2.imwrite(f"{OUT}/{PREFIX}-mouth.png",strip)
print("wrote mouth strip",strip.shape,"crop",CW,"x",CH,"at",(x1,y1))

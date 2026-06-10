#!/usr/bin/env python3
"""Build source|dub side-by-side mp4 with the dub's audio, frame-aligned through the boomerang loop."""
import os, glob, subprocess, sys, json
import numpy as np, cv2
import imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe()
WORK=os.environ.get("WORK","/home/z/work/sw-perf/ml/lipsync-eval/work")
OUT=os.environ.get("OUT", sys.argv[1] if len(sys.argv)>1 else WORK)
PREFIX=os.environ.get("PREFIX","zen-dub")
DESK_AUDIO=os.environ.get("AUDIO", sys.argv[2] if len(sys.argv)>2 else "/mnt/c/Users/z/OneDrive/Desktop/zen-dub-fullnative.mp4")
DUB=os.environ.get("DUB_FRAMES", f"{WORK}/frames_fullnative")
SRC=os.environ.get("SRC_FRAMES", f"{WORK}/frames_source")

dub=sorted(glob.glob(f"{DUB}/*.png"))
src=sorted(glob.glob(f"{SRC}/*.png"))
if not src:
    print("no source frames -> skipping side-by-side video"); sys.exit(0)
src_imgs=[cv2.imread(p) for p in src]; h,w=src_imgs[0].shape[:2]
su=[s[:int(h*0.45)].astype(np.float32) for s in src_imgs]
def align(d):
    u=cv2.imread(d)[:int(h*0.45)].astype(np.float32);b=0;bv=1e18
    for i,s in enumerate(su):
        v=np.mean((u-s)**2)
        if v<bv:bv=v;b=i
    return b

tmp=f"{WORK}/sbs_frames"; os.makedirs(tmp,exist_ok=True)
for f in glob.glob(f"{tmp}/*.png"): os.remove(f)
sep=np.full((h,6,3),(0,255,255),np.uint8)
def lab(img,t,col):
    img=img.copy(); cv2.rectangle(img,(0,0),(img.shape[1],30),(0,0,0),-1)
    cv2.putText(img,t,(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2,cv2.LINE_AA); return img
for k,d in enumerate(dub):
    si=align(d)
    s=lab(src_imgs[si],"SOURCE",(160,255,160))
    dd=lab(cv2.imread(d),"NATIVE DUB",(160,210,255))
    cv2.imwrite(f"{tmp}/{k:05d}.png",np.hstack([s,sep,dd]))
print(f"wrote {len(dub)} sbs frames")

# encode with dub audio
silent=f"{tmp}/_silent.mp4"
subprocess.run([FF,"-y","-hide_banner","-loglevel","error","-r","25","-i",f"{tmp}/%05d.png",
                "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",silent],check=True)
outv=f"{OUT}/{PREFIX}-sidebyside.mp4"
subprocess.run([FF,"-y","-hide_banner","-loglevel","error","-i",silent,"-i",DESK_AUDIO,
                "-c:v","copy","-map","0:v:0","-map","1:a:0","-c:a","aac","-shortest",outv],check=True)
print("wrote",outv, os.path.getsize(outv),"bytes")

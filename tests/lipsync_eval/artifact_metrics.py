#!/usr/bin/env python3
"""Numeric artifact metrics for a dubbed talking-head video.
  - temporal_flicker : mean frame-to-frame abs-diff in the mouth region (0-255). Lower=steadier.
  - seam_strength    : mean gradient magnitude along the BiSeNet/lower-face paste boundary
                       relative to the surrounding facial-skin gradient (ratio; ~1.0 = invisible seam).
  - mouth_sharpness  : Laplacian variance of the mouth crop (higher=sharper; low => VAE softness).
                       Reported both absolute and relative to a non-pasted reference patch (cheek/forehead)
                       so we can isolate VAE softening from the subject's intrinsic sharpness.
Requires frames dir + a mouth bbox json (signal_bbox.json). Self-locates mouth if bbox absent.
"""
import sys, os, glob, json
import numpy as np, cv2

def lap_var(gray):
    # standard sharpness: variance of Laplacian on uint8 gray -> CV_64F
    g8 = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
    return float(cv2.Laplacian(g8, cv2.CV_64F).var())

def align_index(dub_frame, src_upper, h):
    u=dub_frame[:int(h*0.45)].astype(np.float32); b=0; bv=1e18
    for i,s in enumerate(src_upper):
        v=np.mean((u-s)**2)
        if v<bv: bv=v; b=i
    return b

def analyze(frames_dir, bbox=None, sample_every=2, source_dir=None):
    fps_paths=sorted(glob.glob(os.path.join(frames_dir,"*.png")))
    frames=[cv2.imread(p) for p in fps_paths]
    h,w=frames[0].shape[:2]
    if bbox is None:
        cx,cy=int(0.48*w),int(0.375*h)
    else:
        cx,cy=bbox["center"]
    # mouth ROI (the lip-sync write region; generous to include lips+jaw seam)
    MW,MH=int(0.30*w),int(0.22*h)   # ~172x169 on 576x768
    mx1=max(0,cx-MW//2); my1=max(0,cy-MH//2+10); mx2=min(w,mx1+MW); my2=min(h,my1+MH)
    # ---- temporal flicker in mouth region ----
    diffs=[]
    prev=None
    for i in range(0,len(frames),sample_every):
        roi=frames[i][my1:my2, mx1:mx2].astype(np.float32)
        if prev is not None:
            diffs.append(np.mean(np.abs(roi-prev)))
        prev=roi
    flicker=float(np.mean(diffs))
    # also a STATIC-region flicker baseline (forehead, never pasted) to subtract codec noise
    fx1,fy1,fx2,fy2=int(0.40*w),int(0.12*h),int(0.60*w),int(0.20*h)
    bdiffs=[];prev=None
    for i in range(0,len(frames),sample_every):
        roi=frames[i][fy1:fy2, fx1:fx2].astype(np.float32)
        if prev is not None: bdiffs.append(np.mean(np.abs(roi-prev)))
        prev=roi
    flicker_base=float(np.mean(bdiffs))
    # ---- seam strength along lower-face paste boundary ----
    # The paste keeps lower half of a BiSeNet face mask; seam ~ horizontal band near mid-face + jaw arc.
    # Measure gradient on a thin horizontal band at the top edge of the write region vs adjacent skin band.
    seam_ratios=[]; mouth_sharp=[]; ref_sharp=[]
    sample_idx=np.linspace(0,len(frames)-1,40).astype(int)
    for i in sample_idx:
        g=cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
        gm=np.sqrt(gx*gx+gy*gy)
        # seam band: a few rows around the top of the mouth-write region (the upper paste edge)
        sb_y=my1; band=gm[max(0,sb_y-3):sb_y+3, mx1:mx2]
        # reference skin gradient: a band just ABOVE (untouched cheek/skin)
        rb=gm[max(0,sb_y-40):max(0,sb_y-34), mx1:mx2]
        if band.size and rb.size and rb.mean()>1e-3:
            seam_ratios.append(band.mean()/ (rb.mean()+1e-6))
        # sharpness
        mouth_sharp.append(lap_var(g[my1:my2,mx1:mx2]))
        ref_sharp.append(lap_var(g[fy1:fy2,fx1:fx2]))  # forehead (real, never VAE'd)
    out={
        "frames":len(frames),"mouth_roi":[mx1,my1,mx2,my2],
        "temporal_flicker_mouth": round(flicker,3),
        "temporal_flicker_static_baseline": round(flicker_base,3),
        "temporal_flicker_excess": round(flicker-flicker_base,3),
        "seam_strength_ratio": round(float(np.mean(seam_ratios)),3),
        "mouth_sharpness_lapvar": round(float(np.mean(mouth_sharp)),1),
        "ref_sharpness_lapvar_forehead": round(float(np.mean(ref_sharp)),1),
        "sharpness_ratio_mouth_over_forehead": round(float(np.mean(mouth_sharp))/ (np.mean(ref_sharp)+1e-6),3),
    }
    # ---- VAE softness vs SOURCE (the canonical softness test) ----
    # Compare dub mouth sharpness to the original source mouth on aligned frames.
    if source_dir:
        src=[cv2.imread(p) for p in sorted(glob.glob(os.path.join(source_dir,"*.png")))]
        if src:
            su=[s[:int(h*0.45)].astype(np.float32) for s in src]
            ds=[]; ss=[]
            for i in range(0,len(frames),max(sample_every*4,8)):
                si=align_index(frames[i], su, h)
                gd=cv2.cvtColor(frames[i][my1:my2,mx1:mx2],cv2.COLOR_BGR2GRAY)
                gs=cv2.cvtColor(src[si][my1:my2,mx1:mx2],cv2.COLOR_BGR2GRAY)
                ds.append(cv2.Laplacian(gd,cv2.CV_64F).var())
                ss.append(cv2.Laplacian(gs,cv2.CV_64F).var())
            ds_m,ss_m=float(np.mean(ds)),float(np.mean(ss))
            out["dub_mouth_lapvar"]=round(ds_m,1)
            out["source_mouth_lapvar"]=round(ss_m,1)
            out["sharpness_dub_over_source"]=round(ds_m/(ss_m+1e-6),3)
    return out

# ---- thresholds (documented; calibrated against the PyTorch-MuseTalk reference dub) ----
# Reference (zen-dub-demo.mp4, PyTorch MuseTalk): flicker_excess~1.10, seam~2.55, dub/source sharp~?.
THRESH={
 "temporal_flicker_excess": ("<=", 4.0, "mouth flicker minus static-region codec baseline (0-255); PyTorch ref ~1.1; >4 = visible mouth jitter"),
 "seam_strength_ratio":     ("<=", 3.0, "paste-edge gradient / adjacent-skin gradient; ~1=invisible, PyTorch ref ~2.55; >3.0 = native seam worse than reference"),
 "sharpness_dub_over_source": (">=", 0.70, "dub mouth Laplacian-var / SOURCE mouth Laplacian-var on aligned frames; 1.0=matches original, <0.70 = visible VAE softening"),
}
def grade(m):
    out={}
    for k,(op,thr,desc) in THRESH.items():
        if k not in m:  # e.g. source-relative metric when no source provided
            continue
        v=m[k]; ok=(v<=thr) if op=="<=" else (v>=thr)
        out[k]={"value":v,"op":op,"threshold":thr,"pass":bool(ok),"desc":desc}
    return out

if __name__=="__main__":
    fdir=sys.argv[1]; bboxp=sys.argv[2] if len(sys.argv)>2 else None
    src_dir=sys.argv[3] if len(sys.argv)>3 else None
    bbox=json.load(open(bboxp)) if bboxp and os.path.exists(bboxp) else None
    m=analyze(fdir,bbox,source_dir=src_dir)
    m["grades"]=grade(m)
    print(json.dumps(m,indent=2))

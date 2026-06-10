#!/usr/bin/env python3
"""Run canonical syncnet_python (S3FD track + crop + SyncNet) -> LSE-C / LSE-D for a video.
Outputs JSON: {video, lse_c (confidence, higher better), lse_d (min dist, lower better), av_offset, n_tracks}."""
import sys, os, glob, json, subprocess, re, shutil

SP=os.path.dirname(os.path.abspath(__file__))+"/syncnet_python"
FFBIN="/home/z/.venv-lipsync-eval/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
PYBIN="/home/z/.venv-lipsync-eval/bin/python"

def run(video, ref, workdir):
    video=os.path.abspath(video)  # run_pipeline runs with cwd=SP, so abs-path the input
    workdir=os.path.abspath(workdir)
    env=dict(os.environ); env["PATH"]=os.path.dirname(FFBIN)+":"+env.get("PATH","")
    # symlink ffmpeg name into a bin dir on PATH
    bindir=os.path.join(workdir,"_bin"); os.makedirs(bindir,exist_ok=True)
    ln=os.path.join(bindir,"ffmpeg")
    if not os.path.exists(ln): os.symlink(FFBIN, ln)
    env["PATH"]=bindir+":"+env.get("PATH","")
    dd=os.path.join(workdir, ref)
    if os.path.exists(dd): shutil.rmtree(dd)
    os.makedirs(dd)
    # pipeline
    p1=subprocess.run([PYBIN, "run_pipeline.py","--videofile",video,"--reference",ref,
                       "--data_dir",dd,"--min_track","30"], cwd=SP, env=env,
                      capture_output=True, text=True)
    # syncnet
    p2=subprocess.run([PYBIN, "run_syncnet.py","--videofile",video,"--reference",ref,
                       "--data_dir",dd], cwd=SP, env=env, capture_output=True, text=True)
    log=(p1.stdout+p1.stderr+p2.stdout+p2.stderr).replace("\r","\n")
    offs=[int(x) for x in re.findall(r"AV offset:\s*(-?\d+)", log)]
    dists=[float(x) for x in re.findall(r"Min dist:\s*([\d.]+)", log)]
    confs=[float(x) for x in re.findall(r"Confidence:\s*([\d.]+)", log)]
    ntr=len(confs)
    if ntr==0:
        return {"video":os.path.basename(video),"error":"no face track / syncnet result",
                "tail":log.strip().splitlines()[-15:]}
    # report best (max conf) track, and means
    bi=max(range(ntr), key=lambda i:confs[i])
    return {"video":os.path.basename(video),
            "lse_c":round(confs[bi],3),"lse_d":round(dists[bi],3),"av_offset":offs[bi] if offs else None,
            "lse_c_mean":round(sum(confs)/ntr,3),"lse_d_mean":round(sum(dists)/ntr,3),
            "n_tracks":ntr,"all_conf":confs,"all_dist":dists,"all_offset":offs}

if __name__=="__main__":
    video=sys.argv[1]; ref=sys.argv[2] if len(sys.argv)>2 else "v"
    workdir=sys.argv[3] if len(sys.argv)>3 else "/home/z/work/sw-perf/ml/lipsync-eval/work/syncnet"
    os.makedirs(workdir,exist_ok=True)
    r=run(video, ref, workdir)
    print(json.dumps(r,indent=2))

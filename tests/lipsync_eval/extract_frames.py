import sys, os, subprocess, glob
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()

def extract(video, outdir, fps=25, n=None):
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "*.png")): os.remove(f)
    cmd = [FF, "-y", "-hide_banner", "-loglevel", "error", "-i", video,
           "-vf", f"fps={fps}", os.path.join(outdir, "%05d.png")]
    subprocess.run(cmd, check=True)
    frames = sorted(glob.glob(os.path.join(outdir, "*.png")))
    if n: frames = frames[:n]
    return frames

if __name__ == "__main__":
    video, outdir = sys.argv[1], sys.argv[2]
    fr = extract(video, outdir)
    print(f"{len(fr)} frames -> {outdir}")

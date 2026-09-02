import subprocess
from pathlib import Path

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_ghibli_samplers")

for mp4 in sorted(out_dir.glob("*.mp4")):
    gif_out = out_dir / f"{mp4.stem}_animated.gif"
    thumb_out = out_dir / f"{mp4.stem}_thumb.jpg"
    
    # 1. Generate optimized GIF
    cmd_gif = f"ffmpeg -y -i {mp4} -vf 'fps=12,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer' {gif_out}"
    subprocess.run(cmd_gif, shell=True, capture_output=True)
    
    # 2. Extract representative thumbnail
    cmd_thumb = f"ffmpeg -y -ss 00:00:00.600 -i {mp4} -vframes 1 -q:v 2 {thumb_out}"
    subprocess.run(cmd_thumb, shell=True, capture_output=True)
    
    print(f"✓ Processed preview for {mp4.name}")


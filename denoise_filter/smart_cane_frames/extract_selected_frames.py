import cv2, os, csv, shutil
from pathlib import Path

OUT = Path('/mnt/data/smart_cane_frames')
if OUT.exists():
    shutil.rmtree(OUT)
(OUT/'frames').mkdir(parents=True)
(OUT/'contact_sheets').mkdir(parents=True)

videos = {
    'v01_20-16': '/mnt/data/Movie on 13-08-2026 at 20.16.mov',
    'v02_20-18': '/mnt/data/Movie on 13-08-2026 at 20.18.mov',
    'v03_20-20': '/mnt/data/Movie on 13-08-2026 at 20.20(1).mov',
    'v04_20-21': '/mnt/data/Movie on 13-08-2026 at 20.21(1).mov',
    'v05_20-23': '/mnt/data/Movie on 13-08-2026 at 20.23(1).mov',
    'v06_20-24': '/mnt/data/Movie on 13-08-2026 at 20.24(1).mov',
    'v07_20-25': '/mnt/data/Movie on 13-08-2026 at 20.25(2).mov',
}

# Manual coarse selection after reviewing temporally sampled contact sheets.
# Labels are intended as evaluation strata, not absolute physical blur measurements.
selection = {
    'v01_20-16': {
        'clear': [54, 81, 189, 325],
        'moderate_blur': [108, 162, 270, 352, 487],
        'severe_blur': [379, 433, 541],
    },
    'v02_20-18': {
        'clear': [0, 113, 340, 718],
        'moderate_blur': [37, 151, 264, 567, 794],
        'severe_blur': [189, 453, 680],
    },
    'v03_20-20': {
        'clear': [0, 16, 64, 97],
        'moderate_blur': [48, 80, 145, 161, 177],
        'severe_blur': [113, 210, 242],
    },
    'v04_20-21': {
        'clear': [0, 38, 76, 266],
        'moderate_blur': [114, 152, 190, 380, 418],
        'severe_blur': [228, 304, 342],
    },
    'v05_20-23': {
        'clear': [0, 22, 204, 295],
        'moderate_blur': [68, 136, 181, 250, 318],
        'severe_blur': [386, 432, 477],
    },
    'v06_20-24': {
        'clear': [0, 30, 246, 647],
        'moderate_blur': [123, 154, 277, 431, 616],
        'severe_blur': [184, 215, 493],
    },
    'v07_20-25': {
        'clear': [0, 29, 44, 59],
        'moderate_blur': [14, 74, 119, 149, 223],
        'severe_blur': [104, 178, 268],
    },
}

rows=[]
for vid, path in videos.items():
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {path}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for quality, indices in selection[vid].items():
        for idx in indices:
            if idx >= total:
                raise ValueError(f'{vid}: frame {idx} >= total {total}')
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame=cap.read()
            if not ok:
                raise RuntimeError(f'Failed reading {vid} frame {idx}')
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bright=float(gray.mean())
            t=idx/fps
            # Coarse lighting note; v05 deliberately includes a darker tail.
            lighting='normal_indoor'
            if vid=='v05_20-23' and idx>=386:
                lighting='dim_or_occluded'
            fname=f'{vid}_f{idx:05d}_t{t:05.2f}_{quality}.png'
            out=OUT/'frames'/fname
            # PNG is lossless; do not introduce extra JPEG compression into restoration experiments.
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            rows.append({
                'filename': fname,
                'source_video': os.path.basename(path),
                'video_id': vid,
                'frame_number': idx,
                'timestamp_sec': round(t,3),
                'fps': round(fps,6),
                'width': width,
                'height': height,
                'quality_group': quality,
                'lighting_note': lighting,
                'laplacian_variance': round(sharp,3),
                'mean_brightness_0_255': round(bright,3),
            })
    cap.release()

with open(OUT/'metadata.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

readme = '''# Smart Cane Real-Camera Evaluation Frames

This folder contains 84 lossless PNG frames selected from the 7 provided cane-sweep videos (12 frames per video).

## Contents
- `frames/`: flat folder of evaluation images, easy to use with OpenCV/YOLO/annotation tools.
- `metadata.csv`: source video, frame number, timestamp, coarse quality group, lighting note, Laplacian sharpness score, and brightness.
- `contact_sheets/`: overview sheets used to review temporal coverage.
- `extract_selected_frames.py`: reproducible extraction script (paths can be changed for another machine).

## Selection strategy
Frames were intentionally sampled across the video timeline and coarsely grouped as:
- `clear`: relatively informative / less degraded moments
- `moderate_blur`: visible motion degradation but potentially recoverable
- `severe_blur`: challenging motion-blurred frames

The quality groups are coarse evaluation strata, not ground-truth physical blur measurements. `laplacian_variance` is included as an objective sharpness proxy, but it is scene-dependent and should not be treated as a universal blur threshold.

Video `v05_20-23` also contains darker/occluded frames; those are marked `dim_or_occluded` in the metadata.

## Recommended use
1. Annotate only the COCO classes you intend to evaluate (e.g. person, chair, backpack, bottle, dining table).
2. Run the exact same annotated frames through every preprocessing method.
3. Report overall metrics AND metrics by `quality_group`.
4. Keep `raw` as the no-preprocessing control.
5. Do not claim all 84 frames are statistically independent; nearby video frames originate from the same 7 recordings.

PNG is used to avoid introducing extra lossy compression during the restoration experiment.
'''
(OUT/'README.md').write_text(readme)

# Copy contact sheets created during review.
for p in Path('/mnt/data/contact_sheets').glob('*.jpg'):
    shutil.copy2(p, OUT/'contact_sheets'/p.name)

# Copy this script under a friendlier name, but rewrite video paths comments for clarity.
script_src=Path('/mnt/data/build_smart_cane_frames.py')
shutil.copy2(script_src, OUT/'extract_selected_frames.py')

print(f'Created {len(rows)} frames in {OUT}')
from collections import Counter
print('quality counts:', Counter(r['quality_group'] for r in rows))
print('videos:', Counter(r['video_id'] for r in rows))

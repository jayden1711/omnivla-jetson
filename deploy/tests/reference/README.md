# Reference outputs for the setup check

`reference.npz` holds the pose goals and the outputs of the validated deployment (GPTQ int4 LLM + HQQ4 vision on a
Jetson Orin Nano) for 10 FrodoBots-2K frames in pose-goal and image-goal mode. `tools/reference_check.py` requires the
installed runtime to reproduce them bit for bit.

The 20 input images (10 frames, 10 image goals) are not in this repo (FrodoBots-2K, CC BY-SA 4.0, FrodoBots Lab).
`tools/fetch_reference_images.py` downloads them into `frames/` and `goals/`: `sources.json` lists, for each image, the
FrodoBots part, video segment and frame index, and the sha256 of the expected JPEG. `reference_check.py` runs it
automatically when images are missing. It needs ffmpeg; without it on the Jetson, run the script on another machine and
copy the two folders.

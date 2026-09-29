# Reference outputs for the setup check

`reference.npz` holds the pose goals and the outputs of the validated deployment (GPTQ int4 LLM + HQQ4 vision on a
Jetson Orin Nano) for 10 FrodoBots-2K frames in pose-goal (4), image-goal (6), language (7, prompt "move toward the
bench") and language + pose (8, "stay on the path") mode. `tools/reference_check.py` requires the installed runtime to
reproduce them bit for bit.

Recorded 2026-09-28 with `reference_check.py --record-all`, after checking that the runtime with mode-dependent pruning
gives bit-identical outputs to the previous runtime (20/20, modes 4 and 6), and re-checked in a new process (40/40).
The previous file, recorded 2026-09-27 by `eval/jetson/final_validate.py`, differs in one value: image goal, frame
fb_p00_16577_t00182, by 0.0039 action units (about one fp16 rounding step). Neither the old nor the new runtime
reproduces that recorded value (5 runs, with and without warm-up, goal-cache hit and miss); the cause was not found. The
other 19 mode 4/6 outputs are unchanged.

The 20 input images (10 frames, 10 image goals) are not in this repo (FrodoBots-2K, CC BY-SA 4.0, FrodoBots Lab).
`tools/fetch_reference_images.py` downloads them into `frames/` and `goals/`: `sources.json` lists, for each image, the
FrodoBots part, video segment and frame index, and the sha256 of the expected JPEG. `reference_check.py` runs it
automatically when images are missing. It needs ffmpeg; without it on the Jetson, run the script on another machine and
copy the two folders.

# Reference outputs for the setup check

`reference.npz` holds the pose goals and the outputs of the validated deployment (GPTQ int4 LLM + GPTQ int4 Marlin
vision on a Jetson Orin Nano; re-recorded 2026-09-28 after the Marlin-vision gate, then reproduced 40/40 and in 50
fresh processes) for 10 FrodoBots-2K frames in pose-goal (4), image-goal (6), language (7, prompt "move toward the
bench") and language + pose (8, "stay on the path") mode. `tools/reference_check.py` requires the installed runtime to
reproduce them bit for bit.

`reference_cast.npz` holds the outputs of the CAST weights (`OMNIVLA_WEIGHTS=weights_cast`, built from
omnivla-finetuned-cast) on the same frames, goals and prompts, recorded 2026-09-29 and reproduced 40/40 in a new
process; `reference_check.py` uses it automatically for weights whose BUILD_MANIFEST.json names that checkpoint.

`reference_hqq4.npz` holds the previous default's outputs (HQQ4 vision), used by `reference_check.py` and `api_check.py`
when the runtime runs HQQ4 vision (`OMNIVLA_VISION=hqq4`, or weights without `vis_marpc`); the runtime with the Marlin
change reproduced it 40/40 in HQQ4 mode. The rest of this note is about that file:

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

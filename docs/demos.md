# Demo videos

Both demos are open loop: the model predicted paths for recorded video and nothing was driven. They show what the
deployed runtime does and how fast it does it, not how well it drives.

## Online walking footage, on the Jetson (`docs/media/demo_online.gif`, `demo_online.mp4`)

![OmniVLA-7B int4 on a Jetson Orin Nano, open loop on online walking footage](media/demo_online.gif)

The deployed runtime on the Jetson itself, replaying three walking clips from Wikimedia Commons as a live camera: every
prediction takes the frame the video has reached when the previous one finished, so the rate on screen is the real one
(567-571 ms per prediction in language mode, 718 ms with an image goal, on that boot; MAXN SUPER). Language
instructions name objects in view ("move toward the red fence"); one segment uses an image goal. There is no ground
truth for this footage, so it shows behavior and speed, not accuracy. Path: top view in the model's action units.
Made with `eval/demo/online_frames.py`, `eval/jetson/online_demo.py` and `eval/demo/render_online_demo.py`.

Video: Acabashi (CC BY-SA 4.0), Laura Vaara (CC BY 3.0), Pittigrilli (CC0), via Wikimedia Commons; the GIF and MP4
are CC BY-SA 4.0 (credits in the video and in [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)).

## Image goals on FrodoBots-2K clips (`docs/media/demo.gif`, `demo.mp4`)

![Image-goal predictions on FrodoBots-2K clips](media/demo.gif)

Image-goal predictions of the deployed int4 weights on three FrodoBots-2K clips that are not in any test set, one per
1.05 s (the image-goal latency on the Jetson when it was rendered; it is 775 ms now). The predictions were computed on a
GPU with the same weights (they match the Jetson to about 0.003 action units). Blue: predicted path. White: the path
the human driver took. Made with `eval/demo/seq_demo_inputs.py` and `eval/demo/render_demo.py`.

Video: FrodoBots-2K by FrodoBots Lab, CC BY-SA 4.0; the GIF and MP4 are CC BY-SA 4.0 too.

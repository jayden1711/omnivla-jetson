# Third-party licenses and notices

The code in this repository is MIT licensed (see LICENSE). It works with, but does not include, the projects below.
Their licenses apply to those parts.

## OmniVLA (MIT)

https://github.com/NHirose/OmniVLA. Not included: `deploy/setup_jetson.sh` clones it at commit
`5182600cb4a9ee07684e17cdd2a6cbafc56b8a68` and applies `deploy/patches/omnivla.patch`. The patch and the run-time
changes in `deploy/omnivla_deploy.py` modify OmniVLA code, so OmniVLA's notice is reproduced here:

```
MIT License

Original Copyright (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang.
Modifications Copyright (c) 2025 Noriaki Hirose, Catherine Glossop, Dhruv Shah, Sergey Levine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model weights: Llama 2 Community License

OmniVLA's language model is built on Llama 2 7B. The OmniVLA checkpoint (`NHirose/omnivla-original` on Hugging Face) and
the quantized weights produced by `build/build_model.sh` are derivatives of Llama 2 and are distributed under the
Llama 2 Community License Agreement (https://ai.meta.com/llama/license/) and its Acceptable Use Policy. "Llama 2 is
licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved." No weights are
included in this repository.

## Data: FrodoBots-2K (CC BY-SA 4.0)

The evaluation uses frames and logs from FrodoBots-2K by FrodoBots Lab
(https://huggingface.co/datasets/frodobots/FrodoBots-2K), licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Apart from the demo media below, no frames or data files are
included; `eval/data/extract_frames.py`
and the other `eval/data` scripts download the parts they need. Anything you build from the frames (for example the
Kaggle dataset made by `build/make_build_dataset.sh`, or the reference images that
`deploy/tools/fetch_reference_images.py` downloads) must keep the CC BY-SA 4.0 attribution and license.

### Demo media: `docs/media/demo.gif` and `docs/media/demo.mp4` (CC BY-SA 4.0)

These two files are adaptations of FrodoBots-2K video by FrodoBots Lab, licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: three 5-second front-camera clips (rides
`ride_16557_20240117023647`, `ride_17522_20240128045515` and `ride_38564_20240501100413`) were cut, resized and re-encoded,
and overlaid with model predictions, the driven path and text by `eval/demo/render_demo.py`. The two files are licensed
under CC BY-SA 4.0 as well, not MIT. When you share them, keep the attribution: "Contains video from FrodoBots-2K by
FrodoBots Lab (CC BY-SA 4.0); overlays from omnivla-jetson; licensed CC BY-SA 4.0."

## Demo media: `docs/media/demo_online.gif` and `docs/media/demo_online.mp4` (CC BY-SA 4.0)

These two files are adaptations of three videos from Wikimedia Commons, overlaid with predictions that the deployed
model made on a Jetson Orin Nano (`eval/jetson/online_demo.py`, `eval/demo/render_online_demo.py`):

- "Canada Water and Rafter Walk in Southwark, London Docklands" by Acabashi, CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/),
  https://commons.wikimedia.org/wiki/File:Canada_Water_and_Rafter_Walk_in_Southwark,_London_Docklands.webm
- "A Walk Through the Rongankatu Underpass, Tampere" by Laura Vaara (Sounds of Changes), CC BY 3.0
  (https://creativecommons.org/licenses/by/3.0/),
  https://commons.wikimedia.org/wiki/File:A_Walk_Through_the_Rongankatu_Underpass,_Tampere.webm
- "Walk through a seemingly endless cellar below an appartment block in Munich, Germany" by Pittigrilli, CC0 1.0
  (https://creativecommons.org/publicdomain/zero/1.0/),
  https://commons.wikimedia.org/wiki/File:Walk_through_a_seemingly_endless_cellar_below_an_appartment_block_in_Munich,_Germany.webm

Changes: excerpts were cut (segments in `eval/demo/online_demo_plan.json`), center-cropped to a square, resized,
re-encoded without audio, and overlaid with the model's predicted path, the instruction or goal image, the measured
latency and text. Because one source is CC BY-SA 4.0, the two files are licensed under CC BY-SA 4.0 (not MIT); the CC BY
3.0 and CC0 sources allow this. The credits are also shown at the end of the video. When you share them, keep the
attribution above and the license.

## Data: CAST (no license stated)

The language-goal test uses the CAST dataset (https://huggingface.co/datasets/catglossop/CAST-dataset; paper
https://arxiv.org/abs/2508.13446), built on the GNM data mixture. Its dataset card states no license, so nothing from it
is included: `eval/data/cast_extract.py` downloads it when you run it, and `results/` holds only aggregate numbers.

## Data: LeLaN (MIT)

The object-goal language test uses frames and object labels from the LeLaN dataset
(https://huggingface.co/datasets/NHirose/LeLaN_dataset_NoMaD_traj, MIT per its dataset card). Nothing from it is
included: `eval/data/lelan_extract.py` downloads the 210 frames it needs, and `results/` holds only aggregate numbers.

## OmniVLA-edge weights and CLIP

`eval/edge/edge_eval.py` uses the OmniVLA-edge checkpoint (https://huggingface.co/NHirose/omnivla-edge, MIT per its model
card) and OpenAI's CLIP (https://github.com/openai/CLIP, MIT). Neither is included.

## Other code

- Marlin (Apache License 2.0, https://github.com/IST-DASLab/marlin): installed from source at commit 1f25790, not
  included. `deploy/tools/marlin_min_repro.py`, `eval/jetson/marlin_sm87_repro.py` and `docs/marlin_sm87_bug.md` contain
  the function `gen_quant4` copied from Marlin's `test.py`; that function remains under the Apache License 2.0
  (https://www.apache.org/licenses/LICENSE-2.0).
- FrodoBots converter (https://github.com/catglossop/frodo_dataset): `eval/data/fetch_gt_src.sh` downloads three of its
  modules at a pinned commit when `eval/data/gt_extract.py` first runs; no code from that repository is included,
  because it does not state a license.
- OpenVLA-OFT transformers fork (https://github.com/moojink/transformers-openvla-oft, Apache License 2.0), hqq, GemLite,
  timm and the other Python packages in `deploy/requirements-deploy.txt` are installed by pip under their own licenses.

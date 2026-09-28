# Inputs for regenerating the demo predictions with the deployed int4 weights on Kaggle (eval/kaggle/lang_eval.sh with
# LANG_TEST=lelan): every 21st frame of the sequential clips (the image-goal cadence of jetson_cache.py) and each clip's
# goal image, decoded from the same JPEGs the Jetson run used (eval/data/seq_extract.py). Run from the repo root:
#   python eval/demo/seq_demo_inputs.py            -> results/seq_demo.npz
import json
import numpy as np
import pandas as pd
from PIL import Image

S = pd.read_csv("results/seq_frames.csv")
S = S[S.idx % 21 == 0]
arr, meta = {}, {}
for _, r in S.iterrows():
    arr[f"img__{r.frame}"] = np.asarray(Image.open(f"data_seq/frames/{r.frame}.jpg").convert("RGB"), np.uint8)
    meta[r.frame] = r["clip"]
for c in sorted(set(meta.values())):
    arr[f"goal__{c}"] = np.asarray(Image.open(f"data_seq/goals/{c}.jpg").convert("RGB"), np.uint8)
arr["meta"] = np.array(json.dumps(meta))
np.savez_compressed("results/seq_demo.npz", **arr)
print(f"[SEQ] results/seq_demo.npz: {len(meta)} frames, {len(set(meta.values()))} goals")

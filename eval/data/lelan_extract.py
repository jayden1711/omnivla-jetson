# Object-goal language test set from LeLaN (NHirose/LeLaN_dataset_NoMaD_traj), the prompt style OmniVLA's base
# checkpoint was trained on: "move toward <object>". IN-DISTRIBUTION: LeLaN is in omnivla-original's training mix, so this
# checks that language mode works as trained (and survives quantization), not that it generalizes.
# Robot subsets only (Go Stanford gs2 / gs4, SACSoN): object positions are in the robot frame. A frame qualifies if it has
# two labeled objects 0.5-5 m away, in front (x > 0.3 m), whose bearings differ by >= 30 deg and whose phrases share no
# content word (so not "door" vs "wooden door with a window"); one is the target, the
# other the distractor (used by the swapped-prompt control). Prompt = "move toward " + the object's first phrasing, as
# OmniVLA's LeLaN loader builds it. Image = the left 224x224 of the 448x224 frame (the loader's front crop).
# Range requests only (~15 MB). Run from the repo root:  python eval/data/lelan_extract.py [N_PER_SUBSET]
# Output: results/lelan_lang.npz: img__<k> (224x224x3 uint8), and JSON "meta" (target / distractor prompt and position in
# meters, subset, source file, image-shuffle map).
import io, json, math, os, pickle, random, re, sys
import numpy as np
from PIL import Image
from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/NHirose/LeLaN_dataset_NoMaD_traj/resolve/main/dataset_LeLaN_v2.zip"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 70
SUBSETS, SEED, MIN_SEP = ("gs2", "gs4", "sacson"), 0, 30.0

def objects(entries):
    out = []
    for e in entries:
        try:
            x, y = map(float, np.asarray(e["pose_median"]).reshape(-1)[:2]); p = e["prompt"][0][0]
        except Exception:
            continue
        if isinstance(p, str) and p.strip() and x > 0.3 and 0.5 <= math.hypot(x, y) <= 5.0:
            out.append(dict(prompt=p.strip(), xy=[x, y], bearing=math.degrees(math.atan2(y, x))))
    return out

STOP = set("a an the of with and on in at to near its large small big tall long white black gray grey brown blue red "
           "green yellow dark light wooden metal metallic glass".split())
def words(o):
    return {w.strip(".,()'\"") for w in o["prompt"].lower().split()} - STOP - {""}

rng = random.Random(SEED)
with RemoteZip(URL) as z:
    names = [i.filename for i in z.infolist()]
    pk = {s: sorted(n for n in names if f"/dataset_LeLaN_{s}/" in n and n.endswith(".pkl")) for s in SUBSETS}
    have = set(names)
    arr, meta = {}, {}
    for s in SUBSETS:
        cand, got, tried = pk[s][:], 0, 0
        rng.shuffle(cand)
        for p in cand:
            if got >= N or tried >= 40 * N:
                break
            tried += 1
            img = re.sub(r"/pickle_nomad/(\d+)\.pkl$", r"/image/\1.jpg", p)
            if img not in have:
                continue
            obs = objects(pickle.loads(z.read(p)))
            pairs = [(a, b) for a in obs for b in obs if a is not b and not (words(a) & words(b))
                     and abs(a["bearing"] - b["bearing"]) >= MIN_SEP]
            if not pairs:
                continue
            t, d = rng.choice(pairs)
            k = f"{s}_{got:03d}"
            arr[f"img__{k}"] = np.asarray(Image.open(io.BytesIO(z.read(img))).convert("RGB").crop((0, 0, 224, 224)), np.uint8)
            meta[k] = dict(subset=s, file=p, target=t["prompt"], target_xy=t["xy"], distractor=d["prompt"], distractor_xy=d["xy"],
                           folder=os.path.dirname(os.path.dirname(p)))
            got += 1
        print(f"[LELAN] {s}: {got} frames ({tried} pickles read)", flush=True)
keys = sorted(meta)
for k in keys:                                                   # image from another recording, same prompt
    o = [j for j in keys if meta[j]["folder"] != meta[k]["folder"]]
    meta[k]["img_shuffle"] = rng.choice(o)
arr["meta"] = np.array(json.dumps(meta))
os.makedirs("results", exist_ok=True)
np.savez_compressed("results/lelan_lang.npz", **arr)
print(f"[LELAN] wrote results/lelan_lang.npz: {len(keys)} frames")

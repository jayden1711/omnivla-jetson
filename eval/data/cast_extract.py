# Language-goal test set from CAST (catglossop/CAST-dataset, cast_filtered_dataset: GNM trajectories with language
# annotations). The deployed checkpoint (omnivla-original) was not trained on CAST, so the instructions are held out;
# the images come from GNM robots, and GNM is in OmniVLA's training mix.
#
# Ground truth follows OmniVLA's own loader (prismatic/vla/datasets/cast_dataset.py): the 8 states after the current
# one, in the current step's frame, divided by the episode's normalization factor -> action units, (8, 4) x/y/cos/sin.
# One sample per episode, at the first step of the annotated trajectory (the instruction describes it from the start).
# Instruction = the first non-empty of the (up to 10) alternatives, lowercased, as the loader does.
#
# Needs tensorflow + tensorflow_datasets (not on the Jetson). Runs on a Kaggle CPU kernel (downloads ~7 GB):
#   python cast_extract.py OUT.npz [N_EPISODES]
# Output keys: img__<k> (128x128x3 uint8), gt__<k> (8, 4), and a JSON "meta" with per-sample instruction, alternatives,
# source dataset, normalization factor, the image-shuffle and instruction-shuffle maps and the reliability flag.
import glob, json, os, random, sys, tarfile
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "cast_lang.npz"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 240
SEED, MIN_DISP = 0, 1.0          # reliable = final ground-truth displacement >= 1 action unit (fixed before any model run)
WORK = os.environ.get("CAST_WORK", "/tmp/cast")

def to_local_coords(positions, curr_pos, curr_yaw):          # vint_train.data.data_utils (used by cast_dataset.py)
    c, s = np.cos(curr_yaw), np.sin(curr_yaw)
    return (positions - curr_pos).dot(np.array([[c, -s], [s, c]]))

def gt_actions(state, norm):
    pos, yaw = state[:9, :2], state[:9, 2]
    wp = to_local_coords(pos, pos[0], yaw[0])[1:] / norm
    dyaw = yaw[1:] - yaw[0]
    return np.concatenate([wp, np.cos(dyaw)[:, None], np.sin(dyaw)[:, None]], axis=1)

def main():
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from huggingface_hub import hf_hub_download
    tf.config.set_visible_devices([], "GPU")
    if not glob.glob(f"{WORK}/**/features.json", recursive=True):
        tgz = hf_hub_download("catglossop/CAST-dataset", "cast_filtered_dataset.tar.gz", repo_type="dataset", local_dir=WORK)
        with tarfile.open(tgz) as t:
            t.extractall(WORK)
        os.remove(tgz)
    roots = sorted({os.path.dirname(p) for p in glob.glob(f"{WORK}/**/features.json", recursive=True)})
    print(f"[CAST] dataset dirs: {roots}", flush=True)
    rng = random.Random(SEED)
    cand = []
    for root in roots:
        b = tfds.builder_from_directory(root)
        shards = sorted(glob.glob(f"{root}/*.tfrecord*"))
        take = rng.sample(shards, max(1, len(shards) // 4))           # a quarter of the shards, chosen at random
        print(f"[CAST] {root}: {len(shards)} shards, reading {len(take)}", flush=True)
        ds = tf.data.TFRecordDataset(take).map(b.info.features.deserialize_example)
        for ex in ds:
            steps = ex["steps"].batch(100000).get_single_element()
            if not cand:
                print("[CAST] first episode:", tf.nest.map_structure(lambda v: (tuple(v.shape), v.dtype.name), steps),
                      {k: v.numpy() for k, v in ex["episode_metadata"].items()}, flush=True)
            state = steps["observation"]["state"].numpy()
            if len(state) < 10:
                continue
            imgs = steps["observation"]["image"].numpy()
            if imgs[0].mean() > 250:                                  # placeholder white image (counterfactual step)
                continue
            lang = [x.decode() for x in steps["language_instruction"].numpy()[0] if x]
            if not lang:
                continue
            md = {k: v.numpy() for k, v in ex["episode_metadata"].items()}
            norm = float(md.get("normalization_factor", md.get("normalization_fctor")))
            path = md["file_path"].decode() if isinstance(md["file_path"], bytes) else str(md["file_path"])
            if len(cand) < 5:
                print(f"[CAST] path {path} | lang {lang[:3]} | norm {norm} | steps {len(state)}", flush=True)
            cand.append(dict(img=imgs[0], gt=gt_actions(state, norm), lang=lang, norm=norm, path=path,
                             source=path.strip("/").split("/")[-2] if "/" in path else "unknown", n_steps=len(state)))
    print(f"[CAST] candidate episodes: {len(cand)}", flush=True)
    by_src = {}
    for c in cand:
        by_src.setdefault(c["source"], []).append(c)
    pick, srcs = [], sorted(by_src)                                     # round-robin over sources, random within each
    for s in srcs:
        rng.shuffle(by_src[s])
    while len(pick) < N and any(by_src[s] for s in srcs):
        for s in srcs:
            if by_src[s] and len(pick) < N:
                pick.append(by_src[s].pop())
    keys = [f"cast{i:03d}" for i in range(len(pick))]
    # shuffle maps: image from another source; instruction from another episode with a different instruction
    img_shuf, lang_shuf = {}, {}
    for i, k in enumerate(keys):
        o = [j for j in range(len(keys)) if pick[j]["source"] != pick[i]["source"]] or [j for j in range(len(keys)) if j != i]
        img_shuf[k] = keys[rng.choice(o)]
        o = [j for j in range(len(keys)) if pick[j]["lang"][0].lower() != pick[i]["lang"][0].lower()]
        lang_shuf[k] = keys[rng.choice(o)]
    arr, meta = {}, {}
    for k, c in zip(keys, pick):
        arr[f"img__{k}"] = c["img"].astype(np.uint8); arr[f"gt__{k}"] = c["gt"].astype(np.float64)
        meta[k] = dict(instruction=c["lang"][0].lower(), alternatives=c["lang"], source=c["source"], path=c["path"],
                       norm=c["norm"], n_steps=c["n_steps"], reliable=bool(np.linalg.norm(c["gt"][-1, :2]) >= MIN_DISP),
                       img_shuffle=img_shuf[k], lang_shuffle=lang_shuf[k])
    arr["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(OUT, **arr)
    n_rel = sum(m["reliable"] for m in meta.values())
    print(f"[CAST] wrote {OUT}: {len(keys)} samples, {n_rel} reliable, sources "
          + json.dumps({s: sum(m['source'] == s for m in meta.values()) for s in srcs}), flush=True)

if __name__ == "__main__":
    main()

# Split the CAST language samples (cast_extract.py output) into GPTQ calibration and held-out validation, by recording:
# samples cut from the same recording (same path up to "_chunk_") never land on both sides. Per source dataset, the
# recordings are shuffled (seed 0) and taken round-robin over sources until calibration holds >= N_CAL samples; the
# rest is held out. Grouping is global: the same recording can appear in two sources (v1 and v2 of a CAST subset). Fixed before any run on omnivla-finetuned-cast.
#   python eval/data/cast_split.py cast_lang.npz OUT.json [N_CAL=64]
import json, random, re, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
N_CAL = int(sys.argv[3]) if len(sys.argv) > 3 else 64
M = json.loads(str(np.load(src)["meta"]))
rec = lambda k: re.sub(r"_chunk_\d+.*$", "", M[k]["path"].rstrip("/").split("/")[-1])
recs = {}                                                      # recording -> samples (a recording can appear in
for k in sorted(M):                                            # several sources, e.g. lcbc_filtered_dataset and _v2)
    recs.setdefault(rec(k), []).append(k)
srcs = sorted({M[k]["source"] for k in M}); rng = random.Random(0)
order = {s: sorted(r for r, ks in recs.items() if any(M[k]["source"] == s for k in ks)) for s in srcs}
for s in srcs:
    rng.shuffle(order[s])
cal_recs, i = set(), 0
while sum(len(recs[r]) for r in cal_recs) < N_CAL:             # round-robin over sources: next unused recording
    s = srcs[i % len(srcs)]; i += 1
    nxt = next((r for r in order[s] if r not in cal_recs), None)
    if nxt:
        cal_recs.add(nxt)
cal = [k for r in sorted(cal_recs) for k in recs[r]]; held = [k for r in sorted(recs) if r not in cal_recs for k in recs[r]]
assert not {rec(k) for k in cal} & {rec(k) for k in held}, "recording on both sides"
S = dict(calibration=sorted(cal), heldout=sorted(held), n_recordings_cal=len({rec(k) for k in cal}),
         n_recordings_heldout=len({rec(k) for k in held}), heldout_reliable=sum(M[k]["reliable"] for k in held),
         rule="by recording (across sources), round-robin over sources, seed 0")
json.dump(S, open(out, "w"), indent=1)
print(f"[SPLIT] calibration {len(cal)} samples ({S['n_recordings_cal']} recordings) | held-out {len(held)} "
      f"({S['n_recordings_heldout']} recordings, {S['heldout_reliable']} reliable) | recording overlap: none")

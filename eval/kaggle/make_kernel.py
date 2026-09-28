# make_kernel.py SLUG "TITLE" "ENV1 CMD" ["ENV2 CMD" ...] - build kaggle_<slug>/ (notebook + metadata) around kaggle_compress.py.
# Every command runs `python kaggle_compress.py` with the given env prefix. Needs KAGGLE_USER. KERNEL_SOURCES=<you>/<teacher kernel> attaches the
# STUDY=teacher output (labels for STUDY=lora).
import json, os, sys
user = os.environ["KAGGLE_USER"]
slug, title, cmds = sys.argv[1], sys.argv[2], sys.argv[3:]
d = f"kaggle_{slug.replace('-', '_')}"; os.makedirs(d, exist_ok=True)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "kaggle_compress.py")).read()
pip = ("!nvidia-smi --query-gpu=name,memory.total --format=csv\n!pip install -q git+https://github.com/moojink/transformers-openvla-oft.git@"
       "bc339d9ad707454c0c115970db43c260067c61ab tokenizers==0.19.1 timm==0.9.10 accelerate==0.30.1 peft==0.11.1 draccus==0.8.0 "
       "jsonlines json-numpy utm sentencepiece bitsandbytes hqq==0.2.8.post1 pandas")
cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title}"]},
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [pip]},
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["%%writefile /kaggle/working/kaggle_compress.py\n" + src]}]
for c in cmds:
    if c.startswith("!"):                                          # raw shell cell (e.g. an extra pip install)
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [c]}); continue
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [f"!cd /kaggle/working && {c} python kaggle_compress.py 2>&1 | grep -v Warning"]})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(f"{d}/{slug}.ipynb", "w"), indent=1)
json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [os.environ.get("KAGGLE_DATASET", f"{user}/omnivla-frodobots-frames")], "competition_sources": [],
           "kernel_sources": [k for k in os.environ.get("KERNEL_SOURCES", "").split(",") if k], "machine_shape": "NvidiaTeslaT4"}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
print(d)

# Download the deploy weights from Hugging Face into DEST (resumes partial downloads).
# Usage: python tools/download_weights.py REPO_ID REVISION DEST
# The repo may be private: log in first with `hf auth login` or set HF_TOKEN.
import sys
from huggingface_hub import snapshot_download

repo, rev, dest = sys.argv[1:4]
try:
    snapshot_download(repo_id=repo, revision=rev, local_dir=dest)
except Exception as e:
    print(f"[download] {repo}@{rev} failed: {e!r}\n[download] private repo? run `hf auth login` or set HF_TOKEN", flush=True)
    sys.exit(1)
print(f"[download] {repo}@{rev} -> {dest}", flush=True)

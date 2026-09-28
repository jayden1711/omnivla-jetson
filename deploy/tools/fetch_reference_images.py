# Download the 10 reference frames and their image goals from FrodoBots-2K (CC BY-SA 4.0, FrodoBots Lab) into
# tests/reference/{frames,goals}/, using HTTP range requests (only the needed video segments, ~3 MB each) and ffmpeg.
# Every image is checked against the sha256 in tests/reference/sources.json.
# Run: python tools/fetch_reference_images.py   (needs ffmpeg on PATH; exit 0 = all images present and verified)
import hashlib, io, json, os, shutil, subprocess, sys, tempfile, urllib.request, zipfile
import numpy as np
from PIL import Image

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "reference")
W, H = 1024, 576


class HTTPRangeFile:
    """Seekable read-only view of a remote file with a 1 MB block cache (zipfile reads the index and one member at a time)."""
    BLOCK = 1 << 20

    def __init__(self, url):
        self.url, self.pos, self.cache = url, 0, {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            lo, hi = i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        self.cache[i] = r.read()
                    break
                except OSError:
                    if attempt == 2:
                        raise
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            blk = self._block(self.pos // self.BLOCK)
            off = self.pos % self.BLOCK
            chunk = blk[off:off + end - self.pos]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)


def frame_jpeg(ts_path, index):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", ts_path, "-frames:v", str(index + 1), "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    if len(p.stdout) < (index + 1) * W * H * 3:
        raise RuntimeError(f"{ts_path}: fewer than {index + 1} frames")
    a = np.frombuffer(p.stdout, np.uint8, W * H * 3, index * W * H * 3).reshape(H, W, 3)
    b = io.BytesIO()
    Image.fromarray(a).resize((224, 224), Image.BICUBIC).save(b, "JPEG", quality=95)
    return b.getvalue()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    spec = json.load(open(os.path.join(REF, "sources.json")))
    todo = []
    for im in spec["images"]:
        for kind in ("frame", "goal"):
            out = os.path.join(REF, kind + "s", im["name"] + ".jpg")
            if not (os.path.exists(out) and sha(open(out, "rb").read()) == im[kind + "_sha256"]):
                todo.append((im["part"], im[kind + "_segment"], im[kind + "_index"], out, im[kind + "_sha256"]))
    if not todo:
        print("[FETCH] all 20 reference images present and verified")
        return 0
    if shutil.which("ffmpeg") is None:
        print("[FETCH] ffmpeg not found: install it (sudo apt-get install ffmpeg) or run this script on another machine "
              "and copy tests/reference/frames and goals here")
        return 2
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for part in sorted({t[0] for t in todo}):
            url = spec["url"].format(part=part)
            print(f"[FETCH] reading the index of {url}", flush=True)
            with zipfile.ZipFile(HTTPRangeFile(url)) as z:
                for _, seg, index, out, want in [t for t in todo if t[0] == part]:
                    local = os.path.join(tmp, os.path.basename(seg))
                    if not os.path.exists(local):
                        with z.open(seg) as src, open(local, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    data = frame_jpeg(local, index)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(data)
                    ok = sha(data) == want
                    bad += not ok
                    print(f"[FETCH] {os.path.relpath(out, REF)}: {'ok' if ok else 'SHA256 MISMATCH'}", flush=True)
    if bad:
        print(f"[FETCH] {bad} images differ from the reference bytes (different ffmpeg / Pillow / libjpeg build?); "
              "the bit-exact check will fail with them")
        return 1
    print("[FETCH] all 20 reference images downloaded and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

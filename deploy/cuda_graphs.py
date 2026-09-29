"""CUDA graphs for the fixed-shape parts of one OmniVLA prediction (OmniVLADeploy(..., cuda_graphs=True), the default).

What is captured: the two vision encoders (input always 1x3x224x224) and the stack of 32 LLM decoder layers as one graph
per token count (pose goal, image goal and a language prompt length each get their own; at most 3, see install()). One graph for the whole
stack keeps a single input/output buffer per token count: per-layer graphs held 32 of each and ran out of memory on the
8 GB Orin (2026-09-28). Consequence: the per-layer hidden states the LLM returns with output_hidden_states=True are not
meaningful (layers 1-31 pass their input through); the runtime only reads the last one, which is correct. What is not: token selection
(elision and pruning use boolean masks, which need the CPU), the HF mask setup, the projector and the action head;
they run eagerly as before. A graph replays the kernels recorded on the first call with that shape; inputs are copied
into static buffers and outputs are cloned, so callers see ordinary tensors.

Numerics: a replay runs the same kernels on the same inputs, so outputs should be bit-identical to eager mode, but
library code can pick different kernels while capturing (e.g. cuBLAS workspace per stream). Check with
deploy/tools/graph_check.py before relying on it. Goal key/value reuse (goal_refresh > 1) is not supported: its
attention wrapper branches on Python state between calls.
"""
import torch


_SIDE = {}


class _Graphs:
    """one captured graph per input signature; tensors -> static buffers, other arguments must match exactly"""

    def __init__(self, fn, pool, name, n_warmup=1, max_graphs=8):
        self.fn, self.name, self.n_warmup, self.max_graphs = fn, name, n_warmup, max_graphs
        self.pool = pool                                       # callable(device) -> graph memory pool handle
        self.graphs = {}

    @staticmethod
    def _sig(args, kwargs):
        def s(v):
            if torch.is_tensor(v):
                return ("T", tuple(v.shape), v.dtype, str(v.device))
            if v is None or isinstance(v, (bool, int, float, str)):
                return ("V", v)
            return ("O", type(v).__name__)                      # opaque objects (e.g. a KV cache): passed through
        return tuple(s(a) for a in args), tuple((k, s(v)) for k, v in sorted(kwargs.items()))

    def __call__(self, *args, **kwargs):
        key = self._sig(args, kwargs)
        g = self.graphs.get(key)
        if g is None:
            if len(self.graphs) >= self.max_graphs:
                return self.fn(*args, **kwargs)                 # too many shapes: stay eager
            g = self.graphs[key] = self._capture(args, kwargs)
        s_args, s_kwargs, graph, out, dev = g
        with torch.cuda.device(dev):
            for d, v in zip(s_args, args):
                if torch.is_tensor(v):
                    d.copy_(v)
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    s_kwargs[k].copy_(v)
            graph.replay()
            return _clone(out)

    def _capture(self, args, kwargs):
        dev = next(v.device for v in list(args) + list(kwargs.values()) if torch.is_tensor(v))
        with torch.cuda.device(dev):                           # the model may span several GPUs (one graph per layer)
            s_args = [a.clone() if torch.is_tensor(a) else a for a in args]
            s_kwargs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            side = _SIDE.setdefault(str(dev), torch.cuda.Stream())   # one side stream per device: the allocator caches
            side.wait_stream(torch.cuda.current_stream())            # blocks per stream, so a new stream each time leaks
            with torch.cuda.stream(side):                      # warm-up (autotuning, lazy init) outside the capture
                for _ in range(self.n_warmup):
                    self.fn(*s_args, **s_kwargs)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool(dev), stream=side):
                out = self.fn(*s_args, **s_kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.empty_cache()                               # return the warm-up's cached blocks
        return s_args, s_kwargs, graph, out, dev


def _clone(o):
    if torch.is_tensor(o):
        return o.clone()
    if isinstance(o, tuple):
        return tuple(_clone(x) for x in o)
    return o


def install(vla, log=print, max_llm_graphs=3, capture_after=2):
    """wrap the vision encoders and the LLM decoder layers of an OmniVLA model (after the runtime's own patches).
    LLM graphs: one per token count, captured the `capture_after`-th time that count occurs (the ROS node repeats the same
    goal every frame), at most `max_llm_graphs` (pose goal, image goal and one language instruction); other token counts
    run eagerly (bit-identical, ~5% slower). Each LLM graph holds ~40 MB of GPU pool and costs 65-120 MB of RAM, and a
    capture makes that call ~1.5 s: an unbounded cache ran the 8 GB Orin out of memory after ~6 different CAST
    instructions (2026-09-29)."""
    pools = {}                                                  # one pool per (device, purpose): shared by graphs that
    def pool(key):                                              # run one after the other on the same stream
        return lambda dev: pools.setdefault((str(dev), key), torch.cuda.graph_pool_handle())
    vb = vla.vision_backbone
    for name in ("featurizer", "fused_featurizer"):
        mod = getattr(vb, name)
        mod.forward = _Graphs(mod.forward, pool("vision"), f"vision.{name}")
    layers = vla.language_model.model.layers
    orig = [layer.forward for layer in layers]

    def stack(hidden_states, *a, **kw):                        # all decoder layers, no KV cache (the runtime never reads it)
        for f in orig:
            hidden_states = f(hidden_states, *a, **kw)[0]
        return hidden_states
    graphs, seen, warned = {}, {}, []

    def first(hidden_states, *a, **kw):                        # layer 0 runs the whole stack as one graph
        cache = kw.pop("past_key_value", None); use_cache = kw.pop("use_cache", False)
        kw["past_key_value"] = None; kw["use_cache"] = False
        T = hidden_states.shape[1]
        g = graphs.get(T)
        if g is None:
            seen[T] = seen.get(T, 0) + 1
            if seen[T] >= capture_after and len(graphs) < max_llm_graphs:
                g = graphs[T] = _Graphs(stack, pool(("llm", T)), "llm")
            elif len(graphs) >= max_llm_graphs and not warned:
                warned.append(T)
                log(f"[GRAPHS] {max_llm_graphs} LLM graphs in use (token counts {sorted(graphs)}): other token counts "
                    "run without CUDA graphs", flush=True)
        h = g(hidden_states, *a, **kw) if g is not None else stack(hidden_states, *a, **kw)
        return (h, cache) if use_cache else (h,)

    def passthrough(hidden_states, *a, **kw):                  # layers 1..31: already applied by layer 0's graph
        return (hidden_states, kw.get("past_key_value")) if kw.get("use_cache") else (hidden_states,)
    layers[0].forward = first
    for layer in layers[1:]:
        layer.forward = passthrough
    log(f"[GRAPHS] CUDA graphs installed: 2 vision encoders, {len(layers)} LLM layers as one stack "
        f"(vision: first use; LLM: {capture_after}nd use of a token count, at most {max_llm_graphs})", flush=True)

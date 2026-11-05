"""
`exllamav3/modules/moe_lookahead.py` without a GPU, without torch and without numpy's own
scheduling: the prediction path on a stub router, the `EXL3_MOE_LOOKAHEAD_MAX` bound, the refusals,
and the flag-off case, where nothing may run and nothing may be touched.

Why the module is loaded by path rather than by import: importing `exllamav3` executes the package
`__init__`, which needs torch, and this box has none (AGENTS.md: the Linux side of the repo has no
torch). The module under test imports nothing from the engine, so loading its file is the whole of
what is needed -- and it keeps this file collectable by pytest unchanged on the Windows box, where
`python3 tests/moe_cpu/test_lookahead.py` also runs it directly.

What each test pins down is a claim the mechanism makes in its docstring; the one it cannot make
(the prefetch is cheaper than the miss) is a measurement, not a test, and the report says so.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _load(name, relpath):
    path = os.path.join(REPO, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LA = _load("moe_lookahead", os.path.join("exllamav3", "modules", "moe_lookahead.py"))
np = __import__("numpy")


# -- stubs ---------------------------------------------------------------------------------------
# The smallest slice of the torch API the lookahead touches. `FakeTorch.half` is an identity token,
# so `z.dtype != torch.half` is a real comparison and `z.half()` re-tags the tensor once.

HALF = object()


class FakeTorch:
    half = HALF

    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_current_stream_capturing():
            return False


class FakeTensor:
    """Rows of values with the five methods `predict()`/`issue()` call."""

    def __init__(self, rows, dtype = "float"):
        self.rows = [list(r) for r in rows]
        self.shape = (len(self.rows), len(self.rows[0]) if self.rows else 0)
        self.dtype = dtype

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def reshape(self, *shape):
        flat = [v for row in self.rows for v in row]
        if len(shape) == 2 and shape[0] == -1:
            h = shape[1]
            return FakeTensor([flat[i:i + h] for i in range(0, len(flat), h)], self.dtype)
        return self

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def half(self):
        return FakeTensor(self.rows, HALF)

    def detach(self):
        return self

    def to(self, device):
        return self

    def tolist(self):
        return self.rows


class StubRouter:
    """A router that returns `ids` for every row and counts its own calls."""

    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def __call__(self, bsz, cfg, z, params):
        self.calls += 1
        rows = [list(self.ids) for _ in range(bsz)]
        return FakeTensor(rows, dtype = "long"), FakeTensor(rows)


class RecordingNumpy:
    """numpy with a counted `max` (the touch) and an `items` total, so a test can tell a touched
    byte from a claimed one. Dtypes and the rest pass through to the real module."""

    def __init__(self):
        self.calls = 0
        self.items = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def max(self, a):
        self.calls += 1
        self.items += a.size
        return np.max(a)

    def frombuffer(self, *args, **kwargs):
        return np.frombuffer(*args, **kwargs)


class ForbiddenNumpy:
    """A numpy that fails the test if the lookahead reaches for it at all."""

    def max(self, a):
        raise AssertionError("the lookahead touched memory with the flag off")

    def frombuffer(self, *args, **kwargs):
        raise AssertionError("the lookahead mapped the arena with the flag off")


class StubHost:
    def __init__(self, arena_views = None, layer_blocks = None, expert_bytes = 256):
        self.arena_views = [] if arena_views is None else arena_views
        self.layer_blocks = [] if layer_blocks is None else layer_blocks
        self.specs = [dict(expert_bytes = expert_bytes)]


class StubMLP:
    """A split block-sparse module by attribute only (that is all `support_reason`/`cpu_slot` read)."""

    def __init__(self, first = 8, spares = 0, host = None, hidden = 4, ids = (0, 9), **overrides):
        self.key = "model.layers.0.mlp"
        self.routing_gate = object()
        self.tid2eid = None
        self.router_type = "std"
        self.alt_residual_channel = False
        self.cpu_split_first = first
        self._split_spares = spares
        self._split_map = None
        self._split_mp = None
        self.cpu_host = StubHost() if host is None else host
        self.cpu_layer_idx = 0
        self.num_experts = 16
        self.num_experts_per_tok = 2
        self.hidden_size = hidden
        self.mlp_norm = None
        self.router_pre_norm = None
        self.routing_cfg = object()
        self.router = StubRouter(ids)
        self.routing_fn = self.router
        for k, v in overrides.items():
            setattr(self, k, v)


def _arena(blocks = 8, expert_bytes = 256):
    """A host whose one chunk is mapped here, with `blocks` worker slots of `expert_bytes` each."""
    total = blocks * expert_bytes
    view = np.zeros(total // 2, dtype = np.int16)
    layer_blocks = [(0, i * expert_bytes) for i in range(blocks)]
    return StubHost(arena_views = [view], layer_blocks = [layer_blocks],
                    expert_bytes = expert_bytes)


def _layer(mlp = None, max_experts = 8, stats = False, enabled = True, rec = None, host = None):
    """A LayerLookahead over stubs, with its touch counter in `rec`."""
    mlp = mlp if mlp is not None else StubMLP(host = host)
    cfg = LA.LookaheadCFG(enabled = enabled, max_experts = max_experts, stats = stats)
    rec = rec if rec is not None else RecordingNumpy()
    st = LA.LookaheadStats()
    la = LA.LayerLookahead(mlp, cfg, st, toucher = LA.ArenaToucher(numpy_module = rec),
                           layer_idx = 0, torch_module = FakeTorch())
    return la, st, rec


# -- the prediction path -------------------------------------------------------------------------

def test_the_prediction_is_the_layers_own_router_on_the_pre_attention_state():
    # A stub router that always answers (3, 5): the prediction must be exactly that, per row, and it
    # must have gone through the layer's own norms rather than a reimplementation
    order = []

    class Norm:
        def __init__(self, tag):
            self.tag = tag

        def forward(self, z, params, out_dtype = None):
            order.append(self.tag)
            return z

    mlp = StubMLP(hidden = 2, ids = (3, 5))
    mlp.mlp_norm = Norm("mlp_norm")
    mlp.router_pre_norm = Norm("router_pre_norm")
    la, st, rec = _layer(mlp)
    ids = la.predict(FakeTensor([[1.0, 2.0]]))
    assert ids == [[3, 5]], ids
    assert order == ["mlp_norm", "router_pre_norm"], order
    assert mlp.router.calls == 1


def test_the_prefetch_is_the_predicted_host_resident_ids_and_the_bound_holds():
    # 20 host-resident picks predicted, MAX=3: three blocks touched, the first three the router named
    ids = tuple(range(8, 28))
    host = _arena(blocks = 32)
    mlp = StubMLP(first = 8, host = host, ids = ids)
    la, st, rec = _layer(mlp, max_experts = 3)
    touched = la.issue(FakeTensor([[0.5, 0.25]]), {})
    assert touched == 3, touched
    assert rec.calls == 3, rec.calls
    assert la._pred_slots == [0, 1, 2], la._pred_slots
    assert st.experts == 3 and st.nbytes == 3 * 256
    # one line load per 64-byte line of each 256-byte block
    assert rec.items == 3 * 4, rec.items


def test_gpu_resident_picks_are_not_prefetched():
    # ids below cpu_split_first live in VRAM: nothing to fetch, and the step says so by name
    host = _arena()
    mlp = StubMLP(first = 8, host = host, ids = (0, 1, 2, 3))
    la, st, rec = _layer(mlp)
    assert la.issue(FakeTensor([[1.0, 1.0]]), {}) == 0
    assert rec.calls == 0
    assert "no host-resident pick predicted" in st.step_skips


def test_the_bound_is_per_layer_per_step_and_the_router_runs_once():
    host = _arena(blocks = 64)
    mlp = StubMLP(first = 0, host = host, ids = tuple(range(40)))
    la, st, rec = _layer(mlp, max_experts = 8)
    la.issue(FakeTensor([[1.0]]), {})
    la.issue(FakeTensor([[1.0]]), {})
    assert rec.calls == 16, rec.calls            # 8 per step, not 40
    assert mlp.router.calls == 2                 # one prediction per step, not one per id
    assert st.predictions == 2


# -- placement -----------------------------------------------------------------------------------

def test_the_worker_slot_follows_the_placement_map():
    mlp = StubMLP(first = 8)
    # static: slot = id - first, and a GPU-resident id has no slot
    la, _st, _rec = _layer(mlp)
    assert la.cpu_slot(9) == 1 and la.cpu_slot(8) == 0
    assert la.cpu_slot(7) is None
    # dynamic without spares: nothing can move (the promote step needs spares), so the load-time
    # arithmetic still holds
    mlp0 = StubMLP(first = 8, spares = 0)
    mlp0._split_map, mlp0._split_mp = object(), None
    la0, _st0, _rec0 = _layer(mlp0)
    assert la0.cpu_slot(10) == 2
    # dynamic with the host mirror the promote step maintains: expert 6 was promoted into the free
    # GPU slot 0, so its bytes are in VRAM now and nothing should be fetched for it
    mlp2 = StubMLP(first = 8, spares = 2)
    load = [r if r < 6 else 8 + (r - 6) for r in range(16)]
    mlp2._split_mp = list(load)
    la2, _st2, _rec2 = _layer(mlp2)
    assert la2.cpu_slot(10) == 4                              # 8 + (10 - 6) = 12 -> slot 4
    assert la2.cpu_slot(6) == 0                               # still in its worker slot 0
    mlp2._split_mp[6] = 0                                     # promoted: slot 0 is a GPU slot
    assert la2.cpu_slot(6) is None
    # dynamic with spares and no mirror: refuse rather than fetch another expert's block
    mlp3 = StubMLP(first = 8, spares = 2)
    mlp3._split_map = object()
    la3, _st3, _rec3 = _layer(mlp3)
    assert la3.cpu_slot(10) is None


# -- refusals ------------------------------------------------------------------------------------

def test_every_unsupported_layer_is_refused_by_name():
    reasons = {}
    # `hash` and `tid2eid` are one cause reached two ways (a table-selected router): same text
    hash_reason = LA.support_reason(StubMLP(router_type = "sqrtsp_hash"))
    assert hash_reason == LA.support_reason(StubMLP(tid2eid = object()))

    reasons["no gate"] = LA.support_reason(StubMLP(routing_gate = None))
    reasons["hash"] = hash_reason
    reasons["grouped"] = LA.support_reason(StubMLP(router_type = "ds3_grouped"))
    reasons["alt channel"] = LA.support_reason(StubMLP(alt_residual_channel = True))
    reasons["whole-layer offload"] = LA.support_reason(StubMLP(cpu_offload = True))
    reasons["not split"] = LA.support_reason(StubMLP(first = None))
    reasons["no host"] = LA.support_reason(StubMLP(cpu_host = None))
    for name, reason in reasons.items():
        assert reason, f"{name}: a refusal must be named, not None"
    assert len(set(reasons.values())) == len(reasons), reasons      # distinct causes, distinct text
    # and a plain linear router on a split layer is instrumentable
    assert LA.support_reason(StubMLP()) is None


def test_a_refused_layer_is_counted_not_dropped():
    st = LA.LookaheadStats()
    cfg = LA.LookaheadCFG(enabled = True)
    assert LA.install(StubMLP(router_type = "sqrtsp_hash"), cfg = cfg, stats_obj = st) is None
    assert LA.install(StubMLP(first = None), cfg = cfg, stats_obj = st) is None
    summary = st.summary()
    assert summary["layers_attached"] == 0
    assert summary["layers_skipped"] == 2
    assert len(summary["layer_skip_reasons"]) == 2


def test_an_unmapped_arena_is_a_named_refusal_not_a_silent_zero():
    # The served regime: EXL3_MOE_PINNED_ARENA is off, so the parent has no mapping of the worker's
    # arena and cannot touch a byte. That must be counted with the reason, once per layer-step
    host = StubHost(arena_views = [], layer_blocks = [[(0, 0)] * 8])
    mlp = StubMLP(first = 0, host = host, ids = (1, 2))
    la, st, rec = _layer(mlp)
    assert la.issue(FakeTensor([[1.0]]), {}) == 0
    assert rec.calls == 0
    assert st.predictions == 1                       # it did predict; the fetch is what is refused
    assert any("arena is not mapped" in r for r in st.step_skips), st.step_skips
    assert "arena is not mapped" in la.toucher.reason


def test_a_prefill_sized_batch_issues_nothing():
    host = _arena()
    mlp = StubMLP(first = 0, host = host, ids = (1, 2))
    la, st, rec = _layer(mlp)
    x = FakeTensor([[1.0, 2.0, 3.0, 4.0]] * 64)
    assert la.issue(x, {}) == 0
    assert rec.calls == 0 and mlp.router.calls == 0
    assert "prefill-size batch" in st.step_skips
    # a short prompt chunk passes the row gate but is still a prefill: params says so
    assert la.issue(FakeTensor([[1.0, 2.0, 3.0, 4.0]]), {"prefill": True}) == 0
    assert mlp.router.calls == 0
    assert any("prefill pass" in r for r in st.step_skips), st.step_skips


def test_a_captured_graph_turns_the_layer_off_by_name():
    # A device-to-host copy cannot be captured, and running the prediction outside a replayed graph
    # would sync every decode step. The layer must go quiet, and say why, instead of paying either.
    class Capturing:
        half = HALF

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_current_stream_capturing():
                return True

    mlp = StubMLP(first = 0, host = _arena(), ids = (1, 2))
    st = LA.LookaheadStats()
    la = LA.LayerLookahead(mlp, LA.LookaheadCFG(enabled = True), st,
                           toucher = LA.ArenaToucher(numpy_module = RecordingNumpy()),
                           layer_idx = 0, torch_module = Capturing())
    assert la.issue(FakeTensor([[1.0]]), {}) == 0
    assert la.failed and "graph" in la.failed, la.failed
    assert st.layers_skipped == 1 and mlp.router.calls == 0


def test_a_router_that_raises_goes_quiet_after_one_named_failure():
    class Broken:
        def __call__(self, bsz, cfg, z, params):
            raise RuntimeError("kernel refused the pre-attention vector")

    host = _arena()
    mlp = StubMLP(first = 0, host = host, ids = (1, 2))
    mlp.routing_fn = Broken()
    la, st, rec = _layer(mlp)
    assert la.issue(FakeTensor([[1.0]]), {}) == 0
    assert la.failed and "kernel refused" in la.failed
    assert st.layers_skipped == 1
    # and it stays quiet rather than paying the failing call every step
    assert la.issue(FakeTensor([[1.0]]), {}) == 0
    assert st.layers_skipped == 1 and st.predictions == 0


# -- flag off ------------------------------------------------------------------------------------

def test_with_the_flag_off_nothing_runs_and_no_memory_is_touched():
    # The default: EXL3_MOE_LOOKAHEAD unset. install() must return None (so the block's forward sees
    # no hook at all), and a layer driven by hand must not call the router or reach for the arena.
    st = LA.LookaheadStats()
    cfg = LA.LookaheadCFG.from_env({})
    assert cfg.enabled is False
    mlp = StubMLP(host = _arena())
    assert LA.install(mlp, cfg = cfg, stats_obj = st) is None
    assert st.summary()["layers_attached"] == 0 and st.summary()["layers_skipped"] == 0

    la = LA.LayerLookahead(mlp, cfg, LA.LookaheadStats(),
                           toucher = LA.ArenaToucher(numpy_module = ForbiddenNumpy()),
                           layer_idx = 0, torch_module = FakeTorch())
    assert la.issue(FakeTensor([[1.0]]), {}) == 0
    assert mlp.router.calls == 0


def test_the_flags_are_read_from_the_environment_mapping():
    assert LA.LookaheadCFG.from_env({}).enabled is False
    assert LA.LookaheadCFG.from_env({"EXL3_MOE_LOOKAHEAD": "1"}).enabled is True
    assert LA.LookaheadCFG.from_env({"EXL3_MOE_LOOKAHEAD": "0"}).enabled is False
    cfg = LA.LookaheadCFG.from_env({"EXL3_MOE_LOOKAHEAD": "1"})
    assert cfg.max_experts == LA.DEFAULT_MAX_EXPERTS == 8
    assert cfg.stats is False
    cfg = LA.LookaheadCFG.from_env({"EXL3_MOE_LOOKAHEAD": "1", "EXL3_MOE_LOOKAHEAD_MAX": "3",
                                    "EXL3_MOE_LOOKAHEAD_STATS": "1"})
    assert (cfg.max_experts, cfg.stats) == (3, True)
    # a nonsense bound falls back to the default instead of turning the prefetch unbounded or off
    assert LA.LookaheadCFG.from_env({"EXL3_MOE_LOOKAHEAD_MAX": "0"}).max_experts == 8
    assert LA.LookaheadCFG.from_env({"EXL3_MOE_LOOKAHEAD_MAX": "many"}).max_experts == 8


# -- the counter that says whether the mechanism works -------------------------------------------

def test_the_reported_precision_is_the_share_of_prefetched_experts_that_were_routed():
    host = _arena(blocks = 32)
    mlp = StubMLP(first = 8, host = host, ids = (8, 9, 10, 11))
    la, st, rec = _layer(mlp, stats = True)
    la.issue(FakeTensor([[1.0]]), {})
    assert la._pred_router == {8, 9, 10, 11}
    la.note_true(FakeTensor([[9, 10, 0]]))           # two of the four prefetched were routed
    assert (st.checked, st.hits) == (4, 2)
    la.note_true(FakeTensor([[9, 10, 0]]))           # one prediction is scored once
    assert (st.checked, st.hits) == (4, 2)
    text = st.report()
    assert "50.0% of the prefetched experts were routed (2/4)" in text, text
    # with STATS off the comparison is not made at all (it costs a second readback)
    la2, st2, _ = _layer(StubMLP(first = 8, host = _arena()), stats = False)
    la2.issue(FakeTensor([[1.0]]), {})
    la2.note_true(FakeTensor([[9, 10]]))
    assert (st2.checked, st2.hits) == (0, 0)
    assert "not measured" in st2.report()


# -- the engine still calls the hooks -------------------------------------------------------------

def _source(relpath):
    with open(os.path.join(REPO, relpath), "r", encoding = "utf8") as f:
        return f.read()


def test_the_engine_still_calls_the_hooks_the_mechanism_needs():
    # A rename would otherwise surface as "the flag does nothing" on the box, which reads as a
    # measured null. Each string below is a hook or an attribute this module reads.
    transformer = _source(os.path.join("exllamav3", "modules", "transformer.py"))
    assert 'la = getattr(self.mlp, "lookahead", None)' in transformer
    assert "la.issue(x, params)" in transformer

    mlp = _source(os.path.join("exllamav3", "modules", "block_sparse_mlp.py"))
    assert "self.lookahead = None" in mlp
    assert "self.lookahead_install()" in mlp
    assert "self.lookahead.note_true(selected_experts)" in mlp

    cpu = _source(os.path.join("exllamav3", "modules", "block_sparse_mlp_cpu.py"))
    assert "self.cpu_split_first = first" in cpu            # the split boundary cpu_slot() reads
    assert "self._split_spares = spares" in cpu            # and the promotion's spare slots
    assert "m._split_mp = mp" in cpu                       # the host mirror of the placement map

    host = _source(os.path.join("exllamav3", "model", "moe_cpu_host.py"))
    assert "self.arena_views.append(view)" in host          # the parent's mapping of the arena
    assert "self.layer_blocks.append(" in host              # per slot (chunk, offset)
    assert 'spec["expert_bytes"] = gb + ub + db' in host    # the block length
    assert "self.pinned_arena = os.environ.get(\"EXL3_MOE_PINNED_ARENA\", \"0\")" in host


def test_the_probes_predictor_and_this_ones_are_the_same_rule():
    # The 49.7 % was measured with the probe's residual collapse and router re-run; the mechanism
    # states the same rule. If the probe's tuple drifts, the number it measured no longer describes
    # this prediction, so the two lists are held together here.
    probe = _source(os.path.join("bench", "router_lookahead_probe.py"))
    assert 'LINEAR_ROUTER_TYPES = ("std", "std_bias", "dots", "sqrtsp", "ds3")' in probe
    assert tuple(LA.LINEAR_ROUTER_TYPES) == ("std", "std_bias", "dots", "sqrtsp", "ds3")
    assert "x.mean(dim = -2) if (x.dim() == 4 and x.shape[-2] > 1) else x" in probe


# -- runner (pytest free: this box has neither pytest nor torch) ---------------------------------

def _main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                            # noqa: BLE001 - a test runner
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())

#!/usr/bin/env python3
"""Offline gates for the two measurement runners, eval/perf.py and eval/spec_decode.py.

Both runners ship to the Windows box and their stdout becomes a row in passes.csv/arms.csv, so a
defect in the reading or in the comparison writes a wrong number into the results without anything
looking broken. None of that needs a GPU to check:

  * read_prompt() is the single prompt reader, and it decodes the stringified tool-call arguments
    that the `Agentic, *` workloads carry, for the timed and the untimed pass alike;
  * swap_refusal() decides, as a pure function, whether a multi-arm run may run at all while
    dynamic expert placement can move experts between devices mid-generation;
  * verify_task.py's gates: the warm-placement check is per line (an MTP head can be cold while
    every body layer is warm), a quoted ratio band has to match the report's own table, and a
    spec_decode run with no arm row is not a measurement.

Run with: python3 bench/test_gates.py
"""

import ast
import contextlib
import glob
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROMPTS = os.path.join(REPO, "eval", "prompts")
RUNNER = os.path.join(REPO, "eval", "spec_decode.py")

# Importing a runner pulls in torch, tabulate and the installed engine at module scope. The boxes
# that run these gates have none of them (the engine is installed in the Windows venv, not in the
# WSL tree), and none is needed to exercise the pure helpers, so each absent dependency is stubbed.
# Whatever is installed is imported for real, so a run on the box tests the same file it ships.
_STUBS = {
    "torch": {"inference_mode": lambda: (lambda fn: fn),
              "set_printoptions": lambda **kwargs: None},
    "tabulate": {"tabulate": lambda *args, **kwargs: ""},
    "exllamav3": {"Generator": object, "Job": object, "model_init": object,
                  "GreedySampler": object, "TopPSampler": object},
    "exllamav3.generator": {},
    "exllamav3.generator.generator": {"_NO_BATCH_READBACK": False},
    "exllamav3.util": {},
    "exllamav3.util.progress": {"ProgressBar": object},
    "exllamav3.util.misc": {"Timer": object, "cuda_sync_active": object},
    "exllamav3.util.file": {"disk_lru_cache": lambda *args, **kwargs: (lambda fn: fn)},
}


def install_stubs():
    """Make the names a runner imports at module scope resolvable without the GPU stack."""
    for name, attrs in _STUBS.items():
        if name in sys.modules:
            continue
        try:
            __import__(name)
            continue
        except Exception:
            pass
        module = types.ModuleType(name)
        module.__path__ = []  # a stub is reachable as a package for its stubbed children
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
    # a stubbed child has to hang off its parent, which is how `import a.b.c as x` and
    # `from a.b import c` find it
    for name in _STUBS:
        if "." in name:
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, sys.modules[name])


def load_runner(filename):
    """Import a runner from eval/ (it imports the installed engine, not this tree)."""
    install_stubs()
    path = os.path.join(REPO, "eval", filename)
    spec = importlib.util.spec_from_file_location("eval_" + filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spec_decode = load_runner("spec_decode.py")


def load_tool(path, name):
    """Import a bench/ script: they are stdlib-only, so nothing has to be stubbed."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_task = load_tool(os.path.join(HERE, "verify_task.py"), "verify_task_gates")


class ReadPrompt(unittest.TestCase):
    """One prompt reader, shared by the timed and the untimed pass."""

    def test_every_prompt_parses(self):
        # A workload that fails to parse kills the run at that workload, after the earlier ones
        # have already spent the GPU time
        paths = sorted(glob.glob(os.path.join(PROMPTS, "*.json")))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(prompt = os.path.basename(path)):
                data = spec_decode.read_prompt(os.path.basename(path))
                self.assertIsInstance(data, dict)
                self.assertIsInstance(data["messages"], list)

    def test_stringified_arguments_are_decoded(self):
        # The corpus stores tool-call arguments as the JSON string the API sends; agentic_code_01
        # carries no tool calls, so the smallest workload that does is the one under test here.
        # Both the raw file and the reader are checked, so this cannot pass by the corpus changing
        filename = "agentic_code_05.json"
        with open(os.path.join(PROMPTS, filename), "r") as f:
            raw = json.load(f)
        raw_args = [tc["function"]["arguments"]
                    for msg in raw["messages"] for tc in msg.get("tool_calls") or []]
        self.assertTrue(raw_args)
        self.assertIsInstance(raw_args[0], str)

        data = spec_decode.read_prompt(filename)
        args = [tc["function"]["arguments"]
                for msg in data["messages"] for tc in msg.get("tool_calls") or []]
        self.assertEqual(len(args), len(raw_args))
        for a in args:
            self.assertIsInstance(a, dict)

    def test_both_callers_use_the_one_reader(self):
        # The defect was a second copy of the parsing in warm_corpus(); a copy does not fail, it
        # silently templates different input, so the call graph is what is asserted here
        with open(RUNNER, "r") as f:
            tree = ast.parse(f.read())
        for name in ("measure", "warm_corpus"):
            functions = [n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef) and n.name == name]
            self.assertEqual(len(functions), 1, f"{name}() is gone")
            calls = {c.func.id for c in ast.walk(functions[0])
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            self.assertIn("read_prompt", calls, f"{name}() must read prompts through read_prompt()")
            self.assertNotIn("open", calls, f"{name}() must not read the corpus itself")
            self.assertFalse([c for c in ast.walk(functions[0])
                              if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                              and c.func.attr == "load"],
                             f"{name}() must not decode the corpus itself")


class SwapRefusal(unittest.TestCase):
    """The gate itself: a pure decision, so it is testable without a model or a device."""

    def test_one_arm_is_allowed(self):
        # Nothing to compare: dynamic placement cannot make a ratio wrong when there is no ratio
        self.assertIsNone(spec_decode.swap_refusal(1, True))
        self.assertIsNone(spec_decode.swap_refusal(0, True))

    def test_multi_arm_with_dynamic_placement_is_refused(self):
        message = spec_decode.swap_refusal(2, True)
        self.assertIsNotNone(message)
        self.assertIn("EXL3_MOE_CPU_SWAP=0", message)  # the remedy has to be in the text

    def test_multi_arm_with_static_placement_is_allowed(self):
        self.assertIsNone(spec_decode.swap_refusal(2, False))

    def test_arm_count_follows_the_arm_blocks(self):
        # The gate is only as good as its arms argument: a run the harness thinks is single-arm
        # but which measures two would slip through
        args = types.SimpleNamespace(no_baseline = False, s_ngram_match_min = 0,
                                     draft_model_dir = None, readback_ab = False)
        self.assertEqual(spec_decode.count_arms(args), 1)
        args.no_baseline = True
        self.assertEqual(spec_decode.count_arms(args), 0)
        # The n-gram arm is an arm: a baseline + -ngram_min run prints N-gram (greedy) beside
        # Baseline, which is the ratio the gate exists to refuse
        args.no_baseline = False
        args.s_ngram_match_min = 3
        self.assertEqual(spec_decode.count_arms(args), 2)
        args.no_baseline = True
        self.assertEqual(spec_decode.count_arms(args), 1)
        args.no_baseline = False
        args.s_ngram_match_min = 0
        args.draft_model_dir = "/models/draft"
        self.assertEqual(spec_decode.count_arms(args), 2)
        args.readback_ab = True
        self.assertEqual(spec_decode.count_arms(args), 5)

    def test_main_wires_the_gate_in(self):
        # The helpers passing their own tests is not the gate: the block in main() that reads the
        # placement, calls the refusal and exits 2 is what turns them into a refusal, and deleting
        # it once left the whole suite green
        with open(RUNNER, "r") as f:
            tree = ast.parse(f.read())
        functions = [n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "main"]
        self.assertEqual(len(functions), 1, "main() is gone")
        main = functions[0]
        calls = [c for c in ast.walk(main) if isinstance(c, ast.Call)]
        names = {c.func.id for c in calls if isinstance(c.func, ast.Name)}
        self.assertIn("swap_refusal", names, "main() must ask swap_refusal()")
        self.assertIn("dynamic_placement", names, "main() must read the placement, not assume it")
        self.assertIn("count_arms", names, "main() must pass the arm count, not a constant")

        def is_sys_exit(node):
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "exit" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sys")

        exits = [c for c in calls if is_sys_exit(c)]
        self.assertEqual(len(exits), 1, "the refusal path must exit, and exit once")
        # ... under the refusal: deleted guard, or an exit moved out of it, is a run that reports
        # a ratio it should have refused
        guarded = [n for n in ast.walk(main) if isinstance(n, ast.If)
                   and any(isinstance(x, ast.Name) and x.id == "refusal" for x in ast.walk(n.test))
                   and any(is_sys_exit(c) for c in ast.walk(n))]
        self.assertTrue(guarded, "sys.exit must sit under the swap_refusal() result")
        # and the escape hatch has to be consulted, not just documented
        self.assertIn("swap_ok", {n.attr for n in ast.walk(main) if isinstance(n, ast.Attribute)})

    def test_dynamic_placement_reads_the_registry_and_the_modules(self):
        # Placement is fixed at load, so this is a read of two sources: the engine's registry and
        # the modules themselves. Fakes are the whole engine here - no import, no device
        dynamic = types.SimpleNamespace(_split_dynamic = True)
        static = types.SimpleNamespace(_split_dynamic = False)
        def config(registry = None, infer_params = "default"):
            ip = types.SimpleNamespace(moe_cpu_swap_modules = registry) \
                if infer_params == "default" else infer_params
            return types.SimpleNamespace(infer_params = ip)
        self.assertFalse(spec_decode.dynamic_placement(types.SimpleNamespace(modules = []),
                                                       config(registry = [])))
        self.assertFalse(spec_decode.dynamic_placement(types.SimpleNamespace(modules = [static]),
                                                       config(registry = [static])))
        self.assertTrue(spec_decode.dynamic_placement(types.SimpleNamespace(modules = [dynamic]),
                                                      config(registry = [dynamic])))
        # a plain load has no infer_params at all, and a split module is in model.modules even when
        # no registry was ever created
        self.assertFalse(spec_decode.dynamic_placement(types.SimpleNamespace(modules = []),
                                                       types.SimpleNamespace(infer_params = None)))
        self.assertTrue(spec_decode.dynamic_placement(types.SimpleNamespace(modules = [dynamic]),
                                                      types.SimpleNamespace(infer_params = None)))


RATIO_TABLE = """
| band | ratio |
|---|---|
| 4096 | 0.92 |
| 8192 | 0.96 |
| 16384 | 0.99 |
"""


class GatesFixture:
    """A results/ and logs/ tree in a tempdir: the gates only read, so a fixture is the box."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix = "verify-gates-")
        self.addCleanup(shutil.rmtree, self.tmp)
        self.results = os.path.join(self.tmp, "results")
        self.logs = os.path.join(self.tmp, "logs")
        os.makedirs(self.results)
        os.makedirs(self.logs)
        self.passes = os.path.join(self.results, "passes.csv")
        self.write(self.passes, "tag,run_id,status,breach,runner\n")

    def write(self, path, text, newline = None):
        with open(path, "w", encoding = "utf-8", newline = newline) as f:
            f.write(text)

    def add_pass(self, run, runner = "perf", status = "ok", breach = "0"):
        with open(self.passes, "a", encoding = "utf-8") as f:
            f.write(f"tag,{run},{status},{breach},{runner}\n")

    def read_bytes(self, path):
        with open(path, "rb") as f:
            return f.read()

    def add_arm(self, run, arm = "baseline"):
        path = os.path.join(self.results, "arms.csv")
        if not os.path.exists(path):
            self.write(path, "tag,run_id,arm,tok_s,acc,draft,rounds\n")
        with open(path, "a", encoding = "utf-8") as f:
            f.write(f"tag,{run},{arm},45.14,0.00,0.00,1\n")

    def add_log(self, run, lines):
        self.write(os.path.join(self.logs, f"perf-{run}.log"), "\n".join(lines) + "\n")

    def placement(self, count):
        """`count` warm CPU-split lines in the shape the engine prints them."""
        return [f" -- CPU split experts (worker, dynamic): "
                f"model.language_model.layers.{i}.mlp [152..512) of 512, "
                f"placement from saved profile"
                for i in range(count)]

    def report(self, body, run):
        path = os.path.join(self.tmp, "report.md")
        self.write(path, body + f"\nRun `{run}`.\n")
        return path

    def post(self, report):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = verify_task.main(["--post", "--report", report,
                                   "--results", self.results, "--logs", self.logs])
        return rc, buf.getvalue()


class WarmPlacementGate(GatesFixture, unittest.TestCase):
    """Every `CPU split experts` line carries a saved placement, not just one of them."""

    RUN = "mtp-ref-20261030-010101"

    def setUp(self):
        super().setUp()
        self.add_pass(self.RUN)

    def test_every_line_warm_passes(self):
        self.add_log(self.RUN, self.placement(4))
        rc, out = self.post(self.report("# warm\n", self.RUN))
        self.assertEqual(rc, 0, out)
        self.assertIn("4 CPU-split component(s)", out)

    def test_one_cold_component_fails_and_names_it(self):
        # the real shape: 48 warm body layers and a cold MTP head on line 2 of the log, which the
        # old "placement from saved profile" in <whole log> test certified warm
        lines = ([" -- CPU split experts (worker, dynamic): mtp.layers.0.mlp [136..512) of 512"]
                 + self.placement(48))
        self.add_log(self.RUN, lines)
        rc, out = self.post(self.report("# cold head\n", self.RUN))
        self.assertEqual(rc, 1, out)
        self.assertIn("1 of 49", out)
        self.assertIn("mtp.layers.0.mlp", out)

    def test_log_with_no_placement_line_fails(self):
        self.add_log(self.RUN, [" -- Loading model", " -- alloc peak allocated=1 MiB"])
        rc, out = self.post(self.report("# no placement\n", self.RUN))
        self.assertEqual(rc, 1, out)
        self.assertIn("no `CPU split experts` line", out)


class PercentageGate(GatesFixture, unittest.TestCase):
    """A quoted band is checked against the report's own ratio table, not certified by a `%`."""

    RUN = "ratio-20261030-020202"

    def setUp(self):
        super().setUp()
        self.add_pass(self.RUN)
        self.add_log(self.RUN, self.placement(2))

    def test_quote_matching_the_table_passes(self):
        path = self.report("pp at the measurement bands: 0.92–0.99\n" + RATIO_TABLE, self.RUN)
        before = self.read_bytes(path)
        rc, out = self.post(path)
        self.assertEqual(rc, 0, out)
        self.assertIn("matches this report's own ratio table", out)
        # --post is a reader: a gate that rewrites the file it checks is not evidence
        self.assertEqual(self.read_bytes(path), before)

    def test_quote_disagreeing_with_the_table_fails(self):
        path = self.report("pp at the measurement bands (0.95-0.99)\n" + RATIO_TABLE, self.RUN)
        rc, out = self.post(path)
        self.assertEqual(rc, 1, out)
        self.assertIn("does not match", out)
        self.assertIn("0.92", out)
        self.assertIn("0.99", out)

    def test_symmetric_band_fails(self):
        path = self.report("means vary ±5 % run to run\n" + RATIO_TABLE, self.RUN)
        rc, out = self.post(path)
        self.assertEqual(rc, 1, out)
        self.assertIn("symmetric band", out)

    def test_interval_without_a_table_fails(self):
        path = self.report("pp at the measurement bands: 0.92–0.99\n", self.RUN)
        rc, out = self.post(path)
        self.assertEqual(rc, 1, out)
        self.assertIn("quote without a table", out)
        self.assertIn("ratios.py", out)

    def test_plain_percentage_only_informs(self):
        path = self.report("acceptance was 41 % per draft\n", self.RUN)
        rc, out = self.post(path)
        self.assertEqual(rc, 0, out)
        self.assertIn("INFO", out)
        self.assertIn("41 %", out)
        self.assertIn("all gates passed", out)


class ArmRowGate(GatesFixture, unittest.TestCase):
    """A spec_decode run with no arm row died before measuring anything."""

    RUN = "dead-spec-20261030-030303"

    def setUp(self):
        super().setUp()
        self.add_pass(self.RUN, runner = "spec_decode")
        self.add_log(self.RUN, self.placement(2))

    def test_run_without_arm_rows_fails(self):
        self.add_arm("live-spec-20261030-031313")      # the file covers another run, not this one
        rc, out = self.post(self.report("# no arm\n", self.RUN))
        self.assertEqual(rc, 1, out)
        self.assertIn("no row for it in arms.csv", out)

    def test_run_with_arm_rows_passes(self):
        self.add_arm(self.RUN)
        rc, out = self.post(self.report("# arm\n", self.RUN))
        self.assertEqual(rc, 0, out)
        self.assertIn("arm rows present", out)

    def test_spec_decode_csv_alone_is_arm_evidence(self):
        # the archive's older drafting runs print no `ARM` line and exist only in spec_decode.csv,
        # which is why the gate reads both tables instead of whichever one the deploy happened to
        # create first
        self.add_arm("live-spec-20261030-031313")
        self.write(os.path.join(self.results, "spec_decode.csv"),
                   "tag,run_id,category,arm,tok_s,ratio,acc,draft\n"
                   f"tag,{self.RUN},Agentic; code,Baseline,33.84,0,-1,-1\n")
        rc, out = self.post(self.report("# arm\n", self.RUN))
        self.assertEqual(rc, 0, out)
        self.assertIn("arm rows present", out)

    def test_runner_is_read_from_the_overflow_cell(self):
        # the box's passes.csv predates the `runner` column: 20-column header, value appended
        self.add_arm("live-spec-20261030-031313")
        self.write(self.passes, "tag,run_id,status,breach\n")
        with open(self.passes, "a", encoding = "utf-8") as f:
            f.write(f"tag,{self.RUN},ok,0,spec_decode\n")
        rc, out = self.post(self.report("# no arm\n", self.RUN))
        self.assertEqual(rc, 1, out)
        self.assertIn("no row for it in arms.csv", out)


class DeployedStateGate(unittest.TestCase):
    """`--pre` validates the repo state.json; the box loads the deployed copy."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix = "verify-deployed-")
        self.addCleanup(shutil.rmtree, self.tmp)
        self.deployed = os.path.join(self.tmp, "deployed")
        os.makedirs(self.deployed)
        self.repo = {"active": {"model": "D:\\llms\\m", "split": 376, "cache_size": 131072,
                                "cache_quant": "2,2", "draft": {"mode": "mtp"}},
                     "history": []}

    def gate(self, doc = None, text = None):
        path = os.path.join(self.deployed, "state.json")
        if text is not None:
            self.write(path, text, newline = "")
        elif doc is not None:
            self.write(path, json.dumps(doc, indent = 2))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            good = verify_task.deployed_state_gate(self.repo, self.deployed)
        return good, buf.getvalue()

    def write(self, path, text, newline = None):
        with open(path, "w", encoding = "utf-8", newline = newline) as f:
            f.write(text)

    def test_deployed_state_differing_fails(self):
        stale = json.loads(json.dumps(self.repo))
        stale["active"]["split"] = 372
        good, out = self.gate(stale)
        self.assertFalse(good)
        self.assertIn("deploy.sh", out)
        self.assertIn("active.split", out)

    def test_absent_deployed_state_passes(self):
        good, out = self.gate()
        self.assertTrue(good)
        self.assertIn("INFO", out)

    def test_pre_consults_the_deployed_copy(self):
        # the gate is only a gate if --pre asks it: a change edited in the repo and never deployed
        # has to stop the pass even though the repo copy on its own is valid
        state_path = os.path.join(self.tmp, "repo-state.json")
        self.write(state_path, json.dumps(self.repo, indent = 2))
        stale = json.loads(json.dumps(self.repo))
        stale["active"]["split"] = 372
        self.gate(stale)                              # writes the deployed copy
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verify_task.main(["--pre", "--state", state_path, "--deployed", self.deployed])
        self.assertIn("deployed state.json differs", buf.getvalue())

    def test_key_order_and_line_endings_are_not_a_difference(self):
        # core.autocrlf=true writes CRLF on a plain `git checkout -- state.json` while the blob is
        # LF, so the comparison has to be over parsed JSON
        text = json.dumps({"history": [],
                           "active": {"draft": {"mode": "mtp"}, "cache_quant": "2,2",
                                      "cache_size": 131072, "split": 376,
                                      "model": "D:\\llms\\m"}}, indent = 2)
        good, out = self.gate(text = text.replace("\n", "\r\n"))
        self.assertTrue(good, out)
        self.assertIn("PASS", out)


class SizeCheckArgv(unittest.TestCase):
    """`--pre` has to ask about the machine that will be loaded, draft head included."""

    def state(self, draft = {"mode": "mtp"}):
        state = {"model": "D:\\llms\\m", "split": 376, "cache_size": 131072, "cache_quant": "2,2"}
        if draft is not None:
            state["draft"] = draft
        return state

    def test_mtp_regime_asks_for_the_draft_head(self):
        argv = verify_task.size_check_argv(self.state())
        self.assertIn("--draft", argv)
        self.assertEqual(argv[argv.index("--draft") + 1], "mtp")
        self.assertIn("/mnt/d/llms/m", argv)          # the path is still translated for the check
        self.assertEqual(argv.count("--mcs"), 1)

    def test_no_draft_regime_leaves_it_off(self):
        for draft in ({"mode": "none"}, None):        # an old state.json has no draft block
            with self.subTest(draft = draft):
                self.assertNotIn("--draft", verify_task.size_check_argv(self.state(draft)))


if __name__ == "__main__":
    unittest.main(verbosity = 2)

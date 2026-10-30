#!/usr/bin/env python3
"""Offline gates for the two measurement runners, eval/perf.py and eval/spec_decode.py.

Both runners ship to the Windows box and their stdout becomes a row in passes.csv/arms.csv, so a
defect in the reading or in the comparison writes a wrong number into the results without anything
looking broken. None of that needs a GPU to check:

  * read_prompt() is the single prompt reader, and it decodes the stringified tool-call arguments
    that the `Agentic, *` workloads carry, for the timed and the untimed pass alike;
  * swap_refusal() decides, as a pure function, whether a multi-arm run may run at all while
    dynamic expert placement can move experts between devices mid-generation.

Run with: python3 bench/test_gates.py
"""

import ast
import glob
import importlib.util
import json
import os
import sys
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



if __name__ == "__main__":
    unittest.main(verbosity = 2)

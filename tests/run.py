"""Plain test runner (pytest also works): py -3 tests/run.py [name-filter]"""
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main(argv):
    flt = argv[1] if len(argv) > 1 else ""
    passed, failed, skipped = 0, [], []
    t_all = time.perf_counter()
    for fn in sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py")):
        try:
            mod = importlib.import_module(fn[:-3])
        except ModuleNotFoundError as e:
            if e.name != "pytest":
                raise
            print(f"SKIP {fn} (needs pytest; run it with py -3 -m pytest)")
            skipped.append(fn)
            continue
        for name in [n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]:
            if flt and flt not in f"{fn}::{name}":
                continue
            t = time.perf_counter()
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {fn}::{name} ({time.perf_counter() - t:.2f}s)")
            except Exception:
                failed.append(f"{fn}::{name}")
                print(f"FAIL {fn}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} file(s) skipped in {time.perf_counter() - t_all:.1f}s")
    for f in failed:
        print(f"  FAILED {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

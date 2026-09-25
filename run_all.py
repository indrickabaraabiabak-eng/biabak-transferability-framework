"""Run the complete BIABAK Transferability Framework workflow in order (about 3 h on one CPU core with the benchmark extension).
Usage: python run_all.py [--skip-benchmark]"""
import subprocess, sys
STEPS = ["run_step1.py", "postprocess_step1.py", "fix_step1.py", "subset_test_step1.py", "figures_step1.py", "export_step1.py",
         "run_step2.py", "analyze_step2.py", "primary_tests_step2.py", "mlp_seed_check.py", "run_step2b_covariate_sensitivity.py",
         "figures_step2.py", "export_step2.py", "run_step3a_aoa.py", "run_step3b_benchmark.py", "run_step3c_extension.py",
         "analyze_step3.py", "audit_extras.py", "figures_step3.py", "export_step3.py"]
skip = {"run_step3b_benchmark.py", "run_step3c_extension.py"} if "--skip-benchmark" in sys.argv else set()
for s in STEPS:
    if s in skip: continue
    print(f"== {s}", flush=True)
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        sys.exit(f"step {s} failed")
print("workflow complete")

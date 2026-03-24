"""
Hyperparameter sweep for ROCStories training.

Runs multiple training configs in parallel (one per GPU, or sequentially on a single GPU),
then auto-evaluates each checkpoint and logs results to a CSV.

Usage:
    # Run all configs sequentially on one GPU:
    python sweep_rocstories.py

    # Run up to 2 jobs in parallel (e.g., 2 GPUs):
    python sweep_rocstories.py --max_parallel=2

    # Only evaluate existing checkpoints (skip training):
    python sweep_rocstories.py --eval_only

    # Run a specific subset of configs:
    python sweep_rocstories.py --configs=0,1,2
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

# =============================================================================
# SWEEP CONFIGURATIONS
# Each dict overrides the base config (config/train_rocstories.py).
# Every config MUST have a unique 'out_dir'.
# =============================================================================

BASE_CONFIG = "config/train_rocstories.py"

SWEEP_CONFIGS = [
    # --- Dropout sweep ---
    {
        "name": "dropout_0.0",
        "out_dir": "out-sweep/dropout_0.0",
        "dropout": 0.0,
        "max_iters": 5000,
    },
    {
        "name": "dropout_0.1",
        "out_dir": "out-sweep/dropout_0.1",
        "dropout": 0.1,
        "max_iters": 5000,
    },
    {
        "name": "dropout_0.2_baseline",
        "out_dir": "out-sweep/dropout_0.2_baseline",
        "dropout": 0.2,
        "max_iters": 5000,
    },
    {
        "name": "dropout_0.3",
        "out_dir": "out-sweep/dropout_0.3",
        "dropout": 0.3,
        "max_iters": 5000,
    },
    # --- Learning rate sweep (with dropout=0.2) ---
    {
        "name": "lr_6e-4",
        "out_dir": "out-sweep/lr_6e-4",
        "dropout": 0.2,
        "learning_rate": 6e-4,
        "min_lr": 6e-5,
        "max_iters": 5000,
        "lr_decay_iters": 5000,
    },
    {
        "name": "lr_2e-3",
        "out_dir": "out-sweep/lr_2e-3",
        "dropout": 0.2,
        "learning_rate": 2e-3,
        "min_lr": 2e-4,
        "max_iters": 5000,
        "lr_decay_iters": 5000,
    },
    # --- More iterations (to see if we need longer training) ---
    {
        "name": "iters_10000",
        "out_dir": "out-sweep/iters_10000",
        "dropout": 0.2,
        "max_iters": 10000,
        "lr_decay_iters": 10000,
    },
    {
        "name": "iters_15000",
        "out_dir": "out-sweep/iters_15000",
        "dropout": 0.2,
        "max_iters": 15000,
        "lr_decay_iters": 15000,
    },
    # --- Block size (more context) ---
    {
        "name": "block_512",
        "out_dir": "out-sweep/block_512",
        "dropout": 0.2,
        "block_size": 512,
        "batch_size": 32,  # halve batch to keep memory similar
        "max_iters": 5000,
    },
    # --- Batch size ---
    {
        "name": "batch_128_accum",
        "out_dir": "out-sweep/batch_128_accum",
        "dropout": 0.2,
        "batch_size": 64,
        "gradient_accumulation_steps": 2,  # effective batch = 128
        "max_iters": 5000,
    },
    # --- Combined best guesses ---
    {
        "name": "combo_a",
        "out_dir": "out-sweep/combo_a",
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "max_iters": 10000,
        "lr_decay_iters": 10000,
        "block_size": 512,
        "batch_size": 32,
    },
    {
        "name": "combo_b",
        "out_dir": "out-sweep/combo_b",
        "dropout": 0.15,
        "learning_rate": 6e-4,
        "min_lr": 6e-5,
        "max_iters": 10000,
        "lr_decay_iters": 10000,
    },
]


def build_train_cmd(config, gpu_id=0):
    """Build the training command for a config."""
    cmd = [sys.executable, "train.py", BASE_CONFIG]

    # Add all overrides as CLI args
    for k, v in config.items():
        if k == "name":
            continue
        if isinstance(v, bool):
            cmd.append(f"--{k}={'True' if v else 'False'}")
        else:
            cmd.append(f"--{k}={v}")

    # Always disable wandb for sweep runs (avoid conflicts)
    # and always save checkpoint on val improvement
    cmd.append("--wandb_log=False")
    cmd.append("--always_save_checkpoint=False")

    return cmd


def build_eval_cmd(config, gpu_id=0):
    """Build the eval_rocstories.py command for a config."""
    return [
        sys.executable, "eval_rocstories.py",
        f"--init_from=resume",
        f"--out_dir={config['out_dir']}",
    ]


def run_training(config, gpu_id=0):
    """Run a single training job. Returns (name, returncode, duration_seconds)."""
    name = config["name"]
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    cmd = build_train_cmd(config, gpu_id)
    log_path = os.path.join(out_dir, "train.log")

    print(f"\n{'='*60}")
    print(f"[START] {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    duration = time.time() - start

    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{status}] {name} — {duration:.0f}s")

    return name, proc.returncode, duration


def run_eval(config, gpu_id=0):
    """Run eval_ppl.py on a checkpoint. Returns (avg_loss, ppl) or (None, None)."""
    ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP EVAL] {config['name']}: no checkpoint found")
        return None, None

    cmd = build_eval_cmd(config, gpu_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"  [EVAL FAIL] {config['name']}: {result.stderr[-200:]}")
        return None, None

    # Parse output for avg_loss and ppl
    avg_loss = None
    ppl = None
    for line in result.stdout.split("\n"):
        if "avg_loss" in line and ":" in line:
            try:
                avg_loss = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "perplexity" in line and ":" in line:
            try:
                ppl = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    print(f"  [EVAL] {config['name']}: loss={avg_loss}, ppl={ppl}")
    return avg_loss, ppl


def run_parallel(configs, max_parallel, gpu_ids, eval_only=False):
    """Run training jobs with parallelism, then evaluate all."""
    results = []

    if not eval_only:
        # Run training jobs
        active = {}  # gpu_id -> (process, config, start_time, log_file_handle)
        pending = list(enumerate(configs))
        available_gpus = list(gpu_ids[:max_parallel])

        while pending or active:
            # Launch new jobs on available GPUs
            while pending and available_gpus:
                idx, config = pending.pop(0)
                gpu_id = available_gpus.pop(0)
                name = config["name"]
                out_dir = config["out_dir"]
                os.makedirs(out_dir, exist_ok=True)

                cmd = build_train_cmd(config, gpu_id)
                log_path = os.path.join(out_dir, "train.log")

                print(f"\n[START] {name} on GPU {gpu_id}")
                print(f"  cmd: {' '.join(cmd)}")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                log_f = open(log_path, "w")
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
                active[gpu_id] = (proc, config, time.time(), log_f)

            # Check for completed jobs
            for gpu_id in list(active.keys()):
                proc, config, start_time, log_f = active[gpu_id]
                ret = proc.poll()
                if ret is not None:
                    duration = time.time() - start_time
                    log_f.close()
                    status = "OK" if ret == 0 else f"FAIL(rc={ret})"
                    print(f"[{status}] {config['name']} — {duration:.0f}s (GPU {gpu_id})")
                    results.append({
                        "name": config["name"],
                        "returncode": ret,
                        "duration": duration,
                        "config": config,
                    })
                    del active[gpu_id]
                    available_gpus.append(gpu_id)

            if active:
                time.sleep(5)
    else:
        # eval_only mode: just create placeholder results
        for config in configs:
            results.append({
                "name": config["name"],
                "returncode": 0,
                "duration": 0,
                "config": config,
            })

    # Evaluate all checkpoints
    print(f"\n{'='*60}")
    print("EVALUATING ALL CHECKPOINTS")
    print(f"{'='*60}")

    csv_rows = []
    for r in results:
        config = r["config"]
        avg_loss, ppl = run_eval(config, gpu_ids[0])
        r["avg_loss"] = avg_loss
        r["ppl"] = ppl

        csv_rows.append({
            "name": config["name"],
            "ppl": ppl,
            "avg_loss": avg_loss,
            "duration_s": r.get("duration", 0),
            "dropout": config.get("dropout", ""),
            "learning_rate": config.get("learning_rate", ""),
            "max_iters": config.get("max_iters", ""),
            "block_size": config.get("block_size", ""),
            "batch_size": config.get("batch_size", ""),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", ""),
            "out_dir": config["out_dir"],
            "pass": "YES" if ppl is not None and ppl < 25.0 else "NO",
        })

    # Write results CSV
    csv_path = "out-sweep/sweep_results.csv"
    os.makedirs("out-sweep", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print summary
    print(f"\n{'='*60}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Name':<25} {'PPL':>10} {'Loss':>10} {'Time':>8} {'Pass?':>6}")
    print("-" * 65)

    # Sort by PPL (best first)
    sorted_rows = sorted(csv_rows, key=lambda r: r["ppl"] if r["ppl"] is not None else float("inf"))
    for row in sorted_rows:
        ppl_str = f"{row['ppl']:.4f}" if row["ppl"] is not None else "N/A"
        loss_str = f"{row['avg_loss']:.6f}" if row["avg_loss"] is not None else "N/A"
        dur_str = f"{row['duration_s']:.0f}s" if row["duration_s"] else "-"
        print(f"{row['name']:<25} {ppl_str:>10} {loss_str:>10} {dur_str:>8} {row['pass']:>6}")

    print(f"\nResults saved to: {csv_path}")

    # Identify best
    best = sorted_rows[0] if sorted_rows and sorted_rows[0]["ppl"] is not None else None
    if best:
        print(f"\nBest config: {best['name']} — PPL={best['ppl']:.4f}")
        if best["ppl"] < 25.0:
            print(f"  PASSES the PPL < 25.0 target!")
        else:
            print(f"  Still above target. Gap: {best['ppl'] - 25.0:.4f}")
            print(f"  Consider: more iters, lower dropout, larger block_size, or architecture changes.")

    return sorted_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROCStories hyperparameter sweep")
    parser.add_argument("--max_parallel", type=int, default=1,
                        help="Max parallel training jobs (default: 1 = sequential)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing checkpoints, skip training")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated indices of configs to run (e.g., '0,1,2')")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs to use (e.g., '0,1')")
    parser.add_argument("--list", action="store_true",
                        help="List all configs and exit")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    if args.list:
        print("Available sweep configs:")
        for i, cfg in enumerate(SWEEP_CONFIGS):
            print(f"  [{i}] {cfg['name']}: {json.dumps({k:v for k,v in cfg.items() if k not in ('name','out_dir')}, default=str)}")
        sys.exit(0)

    # Select configs
    if args.configs:
        indices = [int(i) for i in args.configs.split(",")]
        configs = [SWEEP_CONFIGS[i] for i in indices]
    else:
        configs = SWEEP_CONFIGS

    print(f"Running {len(configs)} configs, max_parallel={args.max_parallel}, GPUs={gpu_ids}")
    for i, cfg in enumerate(configs):
        print(f"  [{i}] {cfg['name']}")

    run_parallel(configs, args.max_parallel, gpu_ids, eval_only=args.eval_only)

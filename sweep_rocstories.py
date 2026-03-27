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

# BEST_CONFIGS = [
#     {
#         "name": "BEST r2_lr6e4_do015",
#         "out_dir": "out-sweep/r2_lr6e4_do015",
#         "dropout": 0.15,
#         "learning_rate": 6e-4,
#         "min_lr": 6e-5,
#         "max_iters": 5000,
#         "lr_decay_iters": 50000,
#         "eval_interval": 100,
#     },
#     {
#         "name": "BEST r2_wd_02",
#         "out_dir": "out-sweep/r2_wd_02",
#         "dropout": 0.2,
#         "learning_rate": 6e-4,
#         "min_lr": 6e-5,
#         "weight_decay": 0.2,
#         "max_iters": 5000,
#         "lr_decay_iters": 50000,
#         "eval_interval": 100,
#     },
#     {
#         "name": "BEST dropout_0.2_baseline",
#         "out_dir": "out-sweep/dropout_0.2_baseline",
#         "dropout": 0.2,
#         "max_iters": 5000,
#     },
#     {
#         "name": "BEST block_512",
#         "out_dir": "out-sweep/block_512",
#         "dropout": 0.2,
#         "block_size": 512,
#         "batch_size": 32,  # halve batch to keep memory similar
#         "max_iters": 5000,
#     },
#     {
#         "name": "TRY minimum n",
#         "out_dir": "out-sweep/min_n",
#         "dropout": 0.2,
#         "block_size": 512,
#         "batch_size": 32,  # halve batch to keep memory similar
#         "max_iters": 5000,
#         "n_layer": 4,
#         "n_head": 4,
#     },
#     {
#         "name": "BEST minimum n2",
#         "out_dir": "out-sweep/best_min_n",
#         "dropout": 0.2,
#         "block_size": 512,
#         "batch_size": 32,  # halve batch to keep memory similar
#         "max_iters": 5000,
#         "n_layer": 4,
#         "n_head": 4,
#         "n_embd": 256,
#     },
#     {
#         "name": "BEST minimum n2 more iteration",
#         "out_dir": "out-sweep/best_min_n",
#         "dropout": 0.2,
#         "block_size": 512,
#         "batch_size": 32,  # halve batch to keep memory similar
#         "max_iters": 10000,
#         "n_layer": 4,
#         "n_head": 4,
#         "n_embd": 256,
#     },
# ]
SWEEP_CONFIGS = [
    {
        "name": "BEST r2_lr6e4_do015",
        "out_dir": "out-sweep/b-r2_lr6e4_do015",
        "dropout": 0.15,
        "learning_rate": 6e-4,
        "min_lr": 6e-5,
        "max_iters": 50000,
        "lr_decay_iters": 50000,
        "eval_interval": 100,
        "warmup_iters": 1000,
    },
    {
        "name": "BEST r2_wd_02",
        "out_dir": "out-sweep/r2_wd_02",
        "dropout": 0.2,
        "learning_rate": 6e-4,
        "min_lr": 6e-5,
        "weight_decay": 0.2,
        "max_iters": 50000,
        "lr_decay_iters": 50000,
        "eval_interval": 100,
        "warmup_iters": 1000,
    },
    {
        "name": "BEST dropout_0.2_baseline",
        "out_dir": "out-sweep/dropout_0.2_baseline",
        "dropout": 0.2,
        "max_iters": 50000,
        "lr_decay_iters": 50000,
        "warmup_iters": 1000,
    },
    {
        "name": "BEST block_512",
        "out_dir": "out-sweep/block_512",
        "dropout": 0.2,
        "block_size": 512,
        "batch_size": 32,  # halve batch to keep memory similar
        "max_iters": 50000,
        "lr_decay_iters": 50000,
        "warmup_iters": 1000,
    },
    {
        "name": "TRY minimum n",
        "out_dir": "out-sweep/min_n1",
        "dropout": 0.2,
        "block_size": 512,
        "batch_size": 32,  # halve batch to keep memory similar
        "max_iters": 50000,
        "lr_decay_iters": 50000,
        "n_layer": 4,
        "n_head": 4,
        "warmup_iters": 1000,
    },
    {
        "name": "BEST minimum n2",
        "out_dir": "out-sweep/best_min_n2",
        "dropout": 0.2,
        "block_size": 512,
        "batch_size": 32,  # halve batch to keep memory similar
        "max_iters": 50000,
        "lr_decay_iters": 50000,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "warmup_iters": 1000,
    },
    {
        "name": "BEST minimum n2 more iteration",
        "out_dir": "out-sweep/best_min_n3",
        "dropout": 0.2,
        "block_size": 512,
        "batch_size": 32,  # halve batch to keep memory similar
        "max_iters": 50000,
        "lr_decay_iters": 50000,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "warmup_iters": 1000,
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
        if line.strip().startswith("ppl") and ":" in line:
            try:
                ppl = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    print(f"  [EVAL] {config['name']}: loss={avg_loss}, ppl={ppl}")
    return avg_loss, ppl


def _make_csv_row(config, duration, avg_loss, ppl):
    """Build a CSV row dict from a completed run."""
    return {
        "name": config["name"],
        "ppl": ppl,
        "avg_loss": avg_loss,
        "duration_s": duration,
        "dropout": config.get("dropout", ""),
        "learning_rate": config.get("learning_rate", ""),
        "max_iters": config.get("max_iters", ""),
        "block_size": config.get("block_size", ""),
        "batch_size": config.get("batch_size", ""),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", ""),
        "out_dir": config["out_dir"],
        "pass": "YES" if ppl is not None and ppl < 25.0 else "NO",
    }


def _write_csv(csv_rows, csv_path):
    """Write (or overwrite) the results CSV."""
    if not csv_rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)


def _print_summary(csv_rows, csv_path):
    """Print a ranked summary table."""
    print(f"\n{'='*60}")
    print("SWEEP RESULTS SO FAR")
    print(f"{'='*60}")
    print(f"{'Name':<25} {'PPL':>10} {'Loss':>10} {'Time':>8} {'Pass?':>6}")
    print("-" * 65)

    sorted_rows = sorted(csv_rows, key=lambda r: r["ppl"] if r["ppl"] is not None else float("inf"))
    for row in sorted_rows:
        ppl_str = f"{row['ppl']:.4f}" if row["ppl"] is not None else "N/A"
        loss_str = f"{row['avg_loss']:.6f}" if row["avg_loss"] is not None else "N/A"
        dur_str = f"{row['duration_s']:.0f}s" if row["duration_s"] else "-"
        print(f"{row['name']:<25} {ppl_str:>10} {loss_str:>10} {dur_str:>8} {row['pass']:>6}")

    print(f"\nResults saved to: {csv_path}")

    best = sorted_rows[0] if sorted_rows and sorted_rows[0]["ppl"] is not None else None
    if best:
        print(f"\nBest so far: {best['name']} — PPL={best['ppl']:.4f}")
        if best["ppl"] < 25.0:
            print(f"  PASSES the PPL < 25.0 target!")
        else:
            print(f"  Still above target. Gap: {best['ppl'] - 25.0:.4f}")


def run_parallel(configs, max_parallel, gpu_ids, eval_only=False):
    """Run training jobs with parallelism. Evaluate each job as soon as it finishes.

    Multiple jobs can share the same GPU (slot-based parallelism).
    Jobs are assigned to GPUs round-robin across the provided gpu_ids.
    """
    csv_rows = []
    csv_path = "out-sweep/sweep_results.csv"

    if not eval_only:
        active = {}  # slot_id -> (process, config, start_time, log_file_handle, gpu_id)
        pending = list(enumerate(configs))
        available_slots = list(range(max_parallel))

        while pending or active:
            # Launch new jobs on available slots
            while pending and available_slots:
                idx, config = pending.pop(0)
                slot_id = available_slots.pop(0)
                gpu_id = gpu_ids[slot_id % len(gpu_ids)]
                name = config["name"]
                out_dir = config["out_dir"]
                os.makedirs(out_dir, exist_ok=True)

                cmd = build_train_cmd(config, gpu_id)
                log_path = os.path.join(out_dir, "train.log")

                print(f"\n[START] {name} (slot {slot_id}, GPU {gpu_id})")
                print(f"  cmd: {' '.join(cmd)}")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                log_f = open(log_path, "w")
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
                active[slot_id] = (proc, config, time.time(), log_f, gpu_id)

            # Check for completed jobs
            for slot_id in list(active.keys()):
                proc, config, start_time, log_f, gpu_id = active[slot_id]
                ret = proc.poll()
                if ret is not None:
                    duration = time.time() - start_time
                    log_f.close()
                    status = "OK" if ret == 0 else f"FAIL(rc={ret})"
                    print(f"\n[{status}] {config['name']} — {duration:.0f}s (slot {slot_id}, GPU {gpu_id})")

                    # Evaluate immediately
                    avg_loss, ppl = run_eval(config, gpu_id)
                    csv_rows.append(_make_csv_row(config, duration, avg_loss, ppl))
                    _write_csv(csv_rows, csv_path)
                    _print_summary(csv_rows, csv_path)

                    del active[slot_id]
                    available_slots.append(slot_id)

            if active:
                time.sleep(5)
    else:
        # eval_only mode: evaluate all existing checkpoints
        for config in configs:
            avg_loss, ppl = run_eval(config, gpu_ids[0])
            csv_rows.append(_make_csv_row(config, 0, avg_loss, ppl))
            _write_csv(csv_rows, csv_path)
            _print_summary(csv_rows, csv_path)

    # Final summary
    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    _print_summary(csv_rows, csv_path)

    return csv_rows


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

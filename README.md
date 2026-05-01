This is a fork of Danijars Dreamer v3 implementation, bundled with a crafter RL benchmark by the same author (see the submodule). The crafter benchmark is altered to allow texture perturbation. The dreamer code is altered to allow use of crafters texture perturbation during training. Evaluation on a variety of textures is added.  
Find the original readme below. 



# Reproducing the Crafter robustness experiment

Steps to replicate the experiment artefacts:

- `baseline_heatmap.pdf`, `treatment_heatmap.pdf` — mean return per (hue, sat) cell
- `pool_aggregates.csv` — per-(condition, pool) means with 95% bootstrap CIs and paired-diff rows
- `b_vs_t.pdf` — baseline vs treatment line plot across train/test pools
- `achievement_breakdown.csv` — per-achievement success rate by condition × pool, plus Crafter score
- `paired_test_train.txt`, `paired_test_test.txt` — Wilcoxon signed-rank test on per-world differences

## 1. Environment

```bash
git clone <this repo>
cd dreamerv3
docker build -f Treatment.Dockerfile -t textured_dreamer_v3 .
```

Run subsequent steps inside the container. Mount a host directory for
checkpoints and sweep outputs, and expose the GPU:

```bash
docker run --gpus all -v ~/logdir:/logdir textured_dreamer_v3 \
  <command from steps 2–4>
```

Manual install (no Docker) is in the [Manual](#manual) section; once
`pip install -r requirements.txt` succeeds the rest is identical.

## 2. Train the baseline agent

```bash
python dreamerv3/main.py \
  --logdir /logdir/baseline \
  --configs crafter size25m
```

Without the `texture_random` config, the env uses unmodified Crafter
textures (variant 0). Checkpoint: `/logdir/baseline/ckpt`.

## 3. Train the treatment agent

```bash
python dreamerv3/main.py \
  --logdir /logdir/treatment \
  --configs crafter size25m texture_random
```

`texture_random` resamples a train-pool variant on every episode reset
(`env.crafter.texture_variant: train_pool`, `env.crafter.texture_seed: 67`).
Checkpoint: `/logdir/treatment/ckpt`.

## 4. Sweep both checkpoints across all 108 texture variants

Use the same `--world-seed` and `--envs` for both sweeps. The paired
Wilcoxon test in step 5 pairs episodes by `(worker_idx, episode_idx)`,
which only matches across runs when these flags agree.

```bash
python -m embodied.run.eval_texture_sweep \
  --from-checkpoint /logdir/baseline/ckpt/[CHECKPOINT_TIMESTAMP] \
  --outdir /logdir/sweep_baseline \
  --world-seed 67 --envs 4 \
  --variant-ids 0-107 --episodes-per-variant 20 \
  --configs crafter size25m

python -m embodied.run.eval_texture_sweep \
  --from-checkpoint /logdir/treatment/ckpt/[CHECKPOINT_TIMESTAMP] \
  --outdir /logdir/sweep_treatment \
  --world-seed 67 --envs 4 \
  --variant-ids 0-107 --episodes-per-variant 20 \
  --configs crafter size25m texture_random
```

Each call writes `sweep_results.jsonl` (one row per episode) and
`sweep_summary.json` into its `--outdir`.

Make sure there is a `done` file inside the checkpoint folder. This is used internal by dreamer to indicate completeness of the checkpoint. Without it you may see a misleading assertion error claiming the provided path doesn't exist. 

## 5. Run all analysis scripts

```bash
crafter/analysis/run_all.sh \
  /logdir/sweep_baseline/sweep_results.jsonl \
  /logdir/sweep_treatment/sweep_results.jsonl \
  /logdir/analysis_out
```

The wrapper invokes each analysis script in turn and writes the seven
artefacts listed above into `/logdir/analysis_out/`. It calls plain
`python <script>.py`, so it uses whatever interpreter is active. Outside
the container, prefix with `PYTHON="uv run python"` (or
`uv run bash crafter/analysis/run_all.sh ...`) if the local venv is
managed with uv.

If local venv managed by uv use `uv run bash ...` (or prefix with `PYTHON="uv run python"`). 

### Running analysis scripts one by one

The sweep JSONL files are consumed by scripts under `crafter/analysis/`:
`plot_heatmap.py`, `compute_pool_aggregates.py`, `plot_b_vs_t.py`,
`achievement_breakdown.py`, `paired_test.py`. Each is runnable
standalone — see the script's `--help`. To run all five at once on a
baseline/treatment pair, use the wrapper described in the next section.

# 



# Mastering Diverse Domains through World Models

A reimplementation of [DreamerV3][paper], a scalable and general reinforcement
learning algorithm that masters a wide range of applications with fixed
hyperparameters.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

DreamerV3 masters a wide range of domains with a fixed set of hyperparameters,
outperforming specialized methods. Removing the need for tuning reduces the
amount of expert knowledge and computational resources needed to apply
reinforcement learning.

![DreamerV3 Benchmark Scores](https://github.com/danijar/dreamerv3/assets/2111293/0fe8f1cf-6970-41ea-9efc-e2e2477e7861)

Due to its robustness, DreamerV3 shows favorable scaling properties. Notably,
using larger models consistently increases not only its final performance but
also its data-efficiency. Increasing the number of gradient steps further
increases data efficiency.

![DreamerV3 Scaling Behavior](https://user-images.githubusercontent.com/2111293/217356063-0cf06b17-89f0-4d5f-85a9-b583438c98dd.png)

# Instructions

The code has been tested on Linux and Mac and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Scalar metrics are also writting as JSONL files.

# Tips

- All config options are listed in `dreamerv3/configs.yaml` and you can
  override them as flags from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag.
- You can use multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter size50m`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as Scope summaries. Other outputs like WandB and
  TensorBoard can be enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions. Try `--batch_size 1` to rule out an out of memory error.
- Many environments are included, some of which require installing additional
  packages. See the `Dockerfile` for reference.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.



# Disclaimer

This repository contains *a fork* of a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is *very* unrelated to Google or DeepMind at this point. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441

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

# Texture sweep (Crafter robustness experiment)

Evaluate a trained Dreamer agent across the 108 HSV-perturbed texture
variants of Crafter. See `experiment_design.md` for the experiment spec.

Training a treatment agent with per-episode texture randomisation:

```bash
python dreamerv3/main.py --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter size25m texture_random
```

The `texture_random` config entry sets `env.crafter.texture_variant=train_pool`
and `env.crafter.texture_seed=42` on top of the standard crafter stack. The
baseline is trained without this entry.

### Smoke test (run first against the real checkpoint)

```bash
python -m embodied.run.eval_texture_sweep \
  --from-checkpoint /path/to/baseline/ckpt \
  --outdir /tmp/smoke \
  --world-seed 999 \
  --variant-ids 0,53,107 \
  --episodes-per-variant 2 \
  --envs 1 \
  --configs crafter size25m
```

(Pass whatever `--configs ...` stack was used to train the checkpoint —
the sweep must instantiate an agent with the same architecture.)

Expected: `/tmp/smoke/sweep_results.jsonl` with six rows (3 variants × 2
episodes) and `/tmp/smoke/sweep_summary.json` with the run metadata.
Runtime: a few minutes.

### Full sweep

```bash
python -m embodied.run.eval_texture_sweep \
  --from-checkpoint /path/to/ckpt \
  --outdir /path/to/outdir \
  --world-seed <int> \
  --variant-ids 0-107 \
  --episodes-per-variant 20 \
  --envs 4 \
  --configs crafter size25m
```

`--episodes-per-variant` must be divisible by `--envs` (paired-seed invariance
— each worker runs the same number of episodes). `--variant-ids` accepts
comma-separated ints (`0,53,107`), ranges (`0-107`), and the strings
`train_pool` / `test_pool`.

### Analysis

The sweep JSONL files are consumed by scripts under `crafter/analysis/`:
`plot_heatmap.py`, `compute_pool_aggregates.py`, `plot_b_vs_t.py`,
`achievement_breakdown.py`, `paired_test.py`. Each is runnable
standalone — see the script's `--help`.

# Disclaimer

This repository contains a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is unrelated to Google or DeepMind. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441

"""Sweep a trained Dreamer agent across Crafter texture variants.

One JSONL row per completed episode. Designed to be run with a single
checkpoint against a user-specified list of variant IDs.

See experiment_design.md §5 for the spec.
"""

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml


def _ensure_imports_available():
  repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
  for path in (repo_root, repo_root / 'crafter'):
    if str(path) not in sys.path:
      sys.path.insert(0, str(path))


_ensure_imports_available()


def parse_variant_ids(spec):
  from crafter.textures import NUM_VARIANTS, TextureBank
  if spec in ('train_pool',):
    return sorted(TextureBank.TRAIN_POOL)
  if spec in ('test_pool',):
    return sorted(TextureBank.TEST_POOL)
  ids = []
  for token in spec.split(','):
    token = token.strip()
    if not token:
      continue
    if '-' in token:
      lo, hi = token.split('-', 1)
      ids.extend(range(int(lo), int(hi) + 1))
    else:
      ids.append(int(token))
  ids = sorted(set(ids))
  if any(i < 0 or i >= NUM_VARIANTS for i in ids):
    raise ValueError(f'variant ids must be in [0, {NUM_VARIANTS}), got {ids}')
  return ids


def _parse_sweep_args(argv):
  p = argparse.ArgumentParser(add_help=False)
  p.add_argument('--from-checkpoint')
  p.add_argument('--outdir')
  p.add_argument('--world-seed', type=int)
  p.add_argument('--variant-ids', default='0-107')
  p.add_argument('--episodes-per-variant', type=int, default=20)
  p.add_argument('--envs', type=int, default=1)
  p.add_argument('--debug', action='store_true')
  p.add_argument('--random-agent', action='store_true',
                 help='Use RandomAgent instead of loading a checkpoint. '
                      'For smoke-testing the sweep wiring with no GPU.')
  if '-h' in argv or '--help' in argv:
    p.print_help()
    print('\nRequired: --from-checkpoint, --outdir, --world-seed.')
    print('Additional args are forwarded to the Dreamer config parser, '
          'e.g. --configs crafter size25m.')
    sys.exit(0)
  sweep_args, remaining = p.parse_known_args(argv)
  missing = [k for k, v in (
      ('--from-checkpoint', sweep_args.from_checkpoint),
      ('--outdir', sweep_args.outdir),
      ('--world-seed', sweep_args.world_seed)) if v is None]
  if missing and not sweep_args.random_agent:
    p.error(f'missing required args: {missing}')
  if sweep_args.random_agent and sweep_args.from_checkpoint is None:
    sweep_args.from_checkpoint = '<random-agent>'
  if sweep_args.outdir is None:
    p.error('missing required arg: --outdir')
  if sweep_args.world_seed is None:
    p.error('missing required arg: --world-seed')
  return sweep_args, remaining


def _load_config(config_argv):
  """Mirror of dreamerv3.main config resolution."""
  import dreamerv3
  folder = pathlib.Path(dreamerv3.__file__).parent
  raw = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(raw)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(config_argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  return config


def _make_agent(config, random_agent=False):
  from dreamerv3.main import make_env
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  from dreamerv3.agent import Agent
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def _variant_hsv(variant_id):
  from crafter.textures import variant_to_hsv
  h, s, b = variant_to_hsv(variant_id)
  return int(h), float(s), float(b)


def _achievement_names(obs_space):
  return sorted(k[len('log/achievement_'):]
                for k in obs_space if k.startswith('log/achievement_'))


class _EpisodeRecorder:
  """Per-worker running totals. On is_last: append one JSONL row.

  Caps each worker at `max_per_worker` rows so the driver's global
  episode counter can't split unevenly across workers (which would
  break the paired-seed invariant: worker k's i-th episode must always
  use the same world seed across variants).
  """

  def __init__(
      self, out_path, variant_id, world_seed, ach_names,
      max_per_worker, n_workers):
    self._out_path = pathlib.Path(out_path)
    self._out_file = self._out_path.open('a', buffering=1)
    self._variant_id = int(variant_id)
    self._world_seed = int(world_seed)
    self._ach_names = list(ach_names)
    self._max_per_worker = int(max_per_worker)
    self._n_workers = int(n_workers)
    self._returns = defaultdict(float)
    self._lengths = defaultdict(int)
    self._episode_idx = defaultdict(int)
    hue, sat, bright = _variant_hsv(variant_id)
    self._hue, self._sat, self._bright = hue, sat, bright
    self.rows_written = 0

  def __call__(self, tran, worker_idx, **_):
    if tran['is_first']:
      self._returns[worker_idx] = 0.0
      self._lengths[worker_idx] = 0
    self._returns[worker_idx] += float(tran['reward'])
    self._lengths[worker_idx] += 1
    if tran['is_last']:
      if self._episode_idx[worker_idx] >= self._max_per_worker:
        return
      achievements = {
          name: int(tran.get(f'log/achievement_{name}', 0))
          for name in self._ach_names
      }
      row = {
          'variant_id': self._variant_id,
          'episode_idx': self._episode_idx[worker_idx],
          'worker_idx': int(worker_idx),
          'world_seed': self._world_seed,
          'return': round(float(self._returns[worker_idx]), 4),
          'length': int(self._lengths[worker_idx]),
          'hue': self._hue,
          'sat': self._sat,
          'bright': self._bright,
          'achievements': achievements,
      }
      self._out_file.write(json.dumps(row) + '\n')
      self._out_file.flush()
      self._episode_idx[worker_idx] += 1
      self.rows_written += 1

  def all_done(self):
    return all(
        self._episode_idx[k] >= self._max_per_worker
        for k in range(self._n_workers))

  def close(self):
    self._out_file.close()


def run(argv=None):
  sweep_args, config_argv = _parse_sweep_args(argv or sys.argv[1:])
  config = _load_config(config_argv)

  if not config.task.startswith('crafter_'):
    raise ValueError(
        f"config.task must be a crafter task, got {config.task!r}. "
        "Did you forget `--configs crafter size25m` (or whichever stack "
        "matches the trained checkpoint)?")

  variant_ids = parse_variant_ids(sweep_args.variant_ids)
  if sweep_args.episodes_per_variant % sweep_args.envs != 0:
    raise ValueError(
        f'--episodes-per-variant ({sweep_args.episodes_per_variant}) must be '
        f'divisible by --envs ({sweep_args.envs}) for paired-seed invariance.')

  outdir = pathlib.Path(sweep_args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  results_path = outdir / 'sweep_results.jsonl'
  summary_path = outdir / 'sweep_summary.json'

  print(f'Building agent from config...')
  agent = _make_agent(config, random_agent=sweep_args.random_agent)
  if not sweep_args.random_agent:
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(sweep_args.from_checkpoint, keys=['agent'])
    print(f'Loaded checkpoint: {sweep_args.from_checkpoint}')
  else:
    print('Using RandomAgent (no checkpoint).')

  # Infer achievement names from one probe env so the recorder's row shape
  # is stable even if the config is not `logs=True` (we force it below).
  from dreamerv3.main import make_env
  probe = make_env(config, 0, logs=True)
  ach_names = _achievement_names(probe.obs_space)
  probe.close()

  if results_path.exists():
    results_path.unlink()

  policy = lambda *a: agent.policy(*a, mode='eval')
  start = time.time()
  total_rows = 0

  for variant_id in variant_ids:
    print(f'[variant {variant_id}] '
          f'hsv={_variant_hsv(variant_id)} '
          f'worlds=[{sweep_args.world_seed}..+{sweep_args.envs})')
    fns = [
        bind(make_env, config, k,
             texture_variant=variant_id,
             seed=sweep_args.world_seed + k,
             logs=True)
        for k in range(sweep_args.envs)
    ]
    driver = embodied.Driver(fns, parallel=(sweep_args.envs > 1
                                            and not sweep_args.debug))
    max_per_worker = sweep_args.episodes_per_variant // sweep_args.envs
    recorder = _EpisodeRecorder(
        results_path, variant_id, sweep_args.world_seed, ach_names,
        max_per_worker=max_per_worker, n_workers=sweep_args.envs)
    driver.on_step(recorder)
    driver.reset(agent.init_policy)
    try:
      while not recorder.all_done():
        driver(policy, episodes=1)
    finally:
      recorder.close()
      driver.close()
    total_rows += recorder.rows_written
    print(f'[variant {variant_id}] wrote {recorder.rows_written} episodes')

  elapsed = time.time() - start
  summary = {
      'from_checkpoint': str(sweep_args.from_checkpoint),
      'outdir': str(outdir),
      'world_seed': sweep_args.world_seed,
      'variant_ids': variant_ids,
      'episodes_per_variant': sweep_args.episodes_per_variant,
      'envs': sweep_args.envs,
      'total_rows': total_rows,
      'elapsed_seconds': round(elapsed, 1),
      'results_path': str(results_path),
      'random_agent': bool(sweep_args.random_agent),
  }
  summary_path.write_text(json.dumps(summary, indent=2) + '\n')
  print(f'\nDone. {total_rows} rows -> {results_path}')
  print(f'Summary -> {summary_path}')


if __name__ == '__main__':
  run()

"""End-to-end test for the texture-sweep script, using a fake agent.

Exercises: CLI parsing, env kwarg forwarding, per-variant JSONL writing,
episode counting, paired-seed invariance across variants.
"""

import json
import pathlib
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
for _p in (REPO_ROOT, REPO_ROOT / 'crafter'):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))


def _parse_variant_ids_via_module(spec):
  from embodied.run.eval_texture_sweep import parse_variant_ids
  return parse_variant_ids(spec)


def test_parse_variant_ids_single_and_range():
  assert _parse_variant_ids_via_module('0,53,107') == [0, 53, 107]
  assert _parse_variant_ids_via_module('0-3') == [0, 1, 2, 3]
  assert _parse_variant_ids_via_module('0-2,5,10-11') == [0, 1, 2, 5, 10, 11]


def test_parse_variant_ids_pool_names():
  from crafter.textures import TextureBank
  assert _parse_variant_ids_via_module('train_pool') == sorted(
      TextureBank.TRAIN_POOL)
  assert _parse_variant_ids_via_module('test_pool') == sorted(
      TextureBank.TEST_POOL)


def test_parse_variant_ids_rejects_out_of_range():
  with pytest.raises(ValueError):
    _parse_variant_ids_via_module('108')
  with pytest.raises(ValueError):
    _parse_variant_ids_via_module('-1')


def test_sweep_end_to_end_with_random_agent(tmp_path, monkeypatch):
  """3 variants x 2 episodes x 1 env → 6 rows with expected schema."""
  from embodied.run import eval_texture_sweep as sweep

  outdir = tmp_path / 'sweep'
  monkeypatch.setattr(sys, 'argv', [
      'eval_texture_sweep',
      '--random-agent',
      '--outdir', str(outdir),
      '--world-seed', '4242',
      '--variant-ids', '0,36,90',
      '--episodes-per-variant', '2',
      '--envs', '1',
      '--configs', 'crafter',
  ])
  sweep.run()

  results = outdir / 'sweep_results.jsonl'
  summary = outdir / 'sweep_summary.json'
  assert results.exists()
  assert summary.exists()

  rows = [json.loads(line) for line in results.read_text().splitlines()]
  assert len(rows) == 6

  required = {'variant_id', 'episode_idx', 'worker_idx', 'world_seed',
              'return', 'length', 'hue', 'sat', 'bright', 'achievements'}
  for r in rows:
    assert set(r) >= required
    assert r['world_seed'] == 4242
    assert r['variant_id'] in (0, 36, 90)
    assert isinstance(r['achievements'], dict)
    assert len(r['achievements']) == 22

  variants_seen = sorted(set(r['variant_id'] for r in rows))
  assert variants_seen == [0, 36, 90]
  for v in variants_seen:
    ep_idx = sorted(r['episode_idx'] for r in rows if r['variant_id'] == v)
    assert ep_idx == [0, 1]


def test_paired_seed_invariant_across_variants(tmp_path, monkeypatch):
  """With fixed world_seed + deterministic actions, episodes under different
  variants must share identical lengths (same worlds, texture-only diff)."""
  import numpy as np
  from embodied.run import eval_texture_sweep as sweep
  import embodied

  # Monkeypatch RandomAgent to return a deterministic constant-action policy.
  class ConstAgent:
    def __init__(self, obs_space, act_space):
      self.obs_space = obs_space
      self.act_space = act_space

    def init_policy(self, batch_size):
      return ()

    def policy(self, carry, obs, mode='train'):
      batch_size = len(obs['is_first'])
      act = {}
      for k, space in self.act_space.items():
        if k == 'reset':
          continue
        if space.discrete:
          act[k] = np.zeros(batch_size, space.dtype)
        else:
          shape = (batch_size,) + space.shape
          act[k] = np.zeros(shape, space.dtype)
      return carry, act, {}

  monkeypatch.setattr(embodied, 'RandomAgent', ConstAgent)

  outdir = tmp_path / 'sweep_paired'
  monkeypatch.setattr(sys, 'argv', [
      'eval_texture_sweep',
      '--random-agent',
      '--outdir', str(outdir),
      '--world-seed', '7777',
      '--variant-ids', '0,60',
      '--episodes-per-variant', '2',
      '--envs', '1',
      '--configs', 'crafter',
  ])
  sweep.run()

  rows = [json.loads(line)
          for line in (outdir / 'sweep_results.jsonl').read_text().splitlines()]
  by_variant = {}
  for r in rows:
    by_variant.setdefault(r['variant_id'], {})[r['episode_idx']] = r

  # Episode i under variant 0 == episode i under variant 60 (same world,
  # same policy) → identical episode length.
  for ep in (0, 1):
    a = by_variant[0][ep]
    b = by_variant[60][ep]
    assert a['length'] == b['length'], (
        f'paired-seed broken: ep {ep} length {a["length"]} vs {b["length"]}')
    assert a['return'] == b['return'], (
        f'paired-seed broken: ep {ep} return {a["return"]} vs {b["return"]}')


def test_per_worker_episode_cap(tmp_path, monkeypatch):
  """With --envs 2 --episodes-per-variant 4, each worker must finish exactly
  2 episodes per variant — driver's global episode counter cannot split them
  unevenly (e.g. 3+1) and break paired-seed invariance."""
  from embodied.run import eval_texture_sweep as sweep

  outdir = tmp_path / 'sweep_capped'
  monkeypatch.setattr(sys, 'argv', [
      'eval_texture_sweep',
      '--random-agent',
      '--outdir', str(outdir),
      '--world-seed', '1234',
      '--variant-ids', '0,1',
      '--episodes-per-variant', '4',
      '--envs', '2',
      '--debug',  # serial driver, faster + deterministic in tests
      '--configs', 'crafter',
  ])
  sweep.run()

  rows = [json.loads(l)
          for l in (outdir / 'sweep_results.jsonl').read_text().splitlines()]
  per_pair = {}
  for r in rows:
    per_pair.setdefault((r['variant_id'], r['worker_idx']), []).append(
        r['episode_idx'])
  # 2 variants × 2 workers × 2 episodes = 8 rows.
  assert len(rows) == 8, f'got {len(rows)} rows, want 8'
  for (vid, wid), eps in per_pair.items():
    assert sorted(eps) == [0, 1], (
        f'variant {vid} worker {wid} got episode_idx {sorted(eps)}, want [0,1]')

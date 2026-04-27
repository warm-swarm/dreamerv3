import json
import pathlib
import sys

# Use bundled crafter package instead of .env one
_crafter_root = str(pathlib.Path(__file__).resolve().parent.parent.parent / 'crafter')
if _crafter_root not in sys.path:
  sys.path.insert(0, _crafter_root)

import crafter
import elements
import embodied
import numpy as np


class Crafter(embodied.Env):

  def __init__(
      self, task, size=(64, 64), logs=False, logdir=None, seed=None,
      texture_variant=None, texture_seed=None):
    assert task in ('reward', 'noreward')
    # Config-system sentinels: 'none' and -1 map to None (legacy / auto).
    # Pool strings pass through; numeric strings become ints.
    if isinstance(texture_variant, str):
      if texture_variant == 'none':
        texture_variant = None
      elif texture_variant not in ('train_pool', 'test_pool'):
        texture_variant = int(texture_variant)
    if isinstance(texture_seed, int) and texture_seed < 0:
      texture_seed = None
    self._env = crafter.Env(
        size=size, reward=(task == 'reward'), seed=seed,
        texture_variant=texture_variant, texture_seed=texture_seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True

  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._env.observation_space.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
    }
    if self._logs:
      spaces.update({
          f'log/achievement_{k}': elements.Space(np.int32)
          for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    self._reward += reward
    self._length += 1
    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{'log/reward': np.float32(info['reward'] if info else 0.0)},
    )
    if self._logs:
      log_achievements = {
          f'log/achievement_{k}': info['achievements'][k] if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
    return obs

  def _write_stats(self, length, reward, info):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        'texture_variant': self._env.current_variant_id,
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines, mode='w')
    print(f'Wrote stats: {filename}')

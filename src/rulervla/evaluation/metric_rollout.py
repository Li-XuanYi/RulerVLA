import numpy as np


class MetricRolloutSession:
    def __init__(self, policy):
        self.policy = policy
        self.episode_index = None
        self.last_step = None
        self.last_sample = None
        self.action_issued_at = None

    def infer(self, request):
        command = request['command']
        if command == 'reset':
            episode_index = int(request['episode_index'])
            self.policy.reset_episode(episode_index)
            self.episode_index = episode_index
            self.last_step = None
            self.last_sample = None
            self.action_issued_at = None
            return {'state': 'reset', 'episode_index': episode_index}
        if command == 'observe':
            sample = request['sample']
            episode_index = int(sample['episode_index'])
            step = int(sample['frame_index'])
            if episode_index != self.episode_index:
                raise ValueError('Episode reset is required before a new rollout')
            expected = 0 if self.last_step is None else self.last_step + 1
            if step != expected:
                raise ValueError('Observe every environment control step in order')
            self.policy.observe_control_step(sample)
            self.last_step = step
            self.last_sample = sample
            return {'state': 'observed', 'episode_index': episode_index, 'control_step': step}
        if command == 'act':
            if self.last_sample is None or self.action_issued_at == self.last_step:
                raise ValueError('One current observation is required per action query')
            prediction = self.policy.infer(self.last_sample)
            actions = np.asarray(prediction['actions'])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError('Policy action chunk must be finite [10,7]')
            self.action_issued_at = self.last_step
            return {'actions': actions, 'episode_index': self.episode_index,
                    'control_step': self.last_step}
        raise ValueError('Unknown Metric Cue rollout command')

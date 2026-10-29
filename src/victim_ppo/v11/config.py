from dataclasses import dataclass
from pathlib import Path
import yaml

ARMS = ('ppo_v11_vix_nav', 'ppo_v11_vix_mix', 'ppo_v11_fixed_nav')
FOLDS = {
    'fold_1': ('2014-12-31', '2015-01-01', '2016-12-31'),
    'fold_2': ('2016-12-31', '2017-01-01', '2018-12-31'),
    'fold_3': ('2018-12-31', '2019-01-01', '2021-12-31'),
    'known': ('2021-12-31', '2022-01-01', '2025-12-31'),
}

@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict

    @property
    def model(self):
        return self.raw['model']

def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    expected = {
        'version': 'v11spec2_200k', 'policy': 'MlpPolicy', 'device': 'cpu',
        'initial_requested_transitions': 200000, 'initial_expected_transitions': 200704,
        'fine_tune_transitions': 32768, 'initialization_sessions': 252,
        'episode_transitions': 63, 'half_life': 126, 'recent_mass': .8,
        'vix_reference': 20., 'vix_exponent': 2, 'period_normal': 126,
        'period_min': 21, 'period_max': 252, 'vix_fill_limit': 3,
        'weights': [.3, .2, .5], 'beta': .1, 'streak_cap': 5,
    }
    model = dict(learning_rate_initial=.0003, learning_rate_final=.00003,
                 n_steps=2048, batch_size=64, n_epochs=10, gamma=.99,
                 gae_lambda=.95, clip_range=.2, normalize_advantage=True,
                 ent_coef=0., vf_coef=.5, max_grad_norm=.5,
                 policy_net_arch=[64,64], activation_fn='Tanh')
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f'frozen V11 contract changed: {key}')
    if raw['model'] != model:
        raise ValueError('frozen PPO hyperparameters changed')
    return Config(path, raw)

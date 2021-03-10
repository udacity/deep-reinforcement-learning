import numpy as np
import pandas as pd

def get_noise_schedulling(n_episodes, decay, noise):
    noise_samples = dict()

    for i in range(n_episodes):
        noise_raw = noise.sample()[0]
        decay_coef = decay ** i
        noise_dec = noise_raw * decay_coef

        noise_samples[i] = [decay_coef, noise_raw, noise_dec]
    res = pd.DataFrame.from_dict(noise_samples, orient='index', columns=['decay', 'noise_raw', 'noise_dec'])
    res['decay_weight'] = decay

    return res


def action_scaler_fn(actions, lower: float = -1., upper: float = 1.):
    """Clips action values between lower and upper"""

    return np.clip(actions, lower, upper)
""" Function to support the dynamic pricing algorithm """
import secrets
import numpy as np

def prepare_samples(
        sample_size: int,
        means: list or tuple,
        st_devs: list or tuple,
        seed: int = 123,
        descending: bool = False
):
    """
    Prepare behavioural samples to create a discrete distribution
    instead of the continuous normal
    :param sample_size: numer of samples per class
    :param means: means per each class
    :param st_devs: respective standard deviations
    :param seed: seed for reproducibility
    :param descending: return values in descending order
    :return: discrete behavioural samples
    """
    rng = np.random.default_rng(secrets.randbits(seed))
    out = []

    for subpop_num, mean in enumerate(means):
        pop_sample = rng.normal(
            loc=mean,
            scale=st_devs[subpop_num],
            size=sample_size
        )
        pop_sample = [(t, subpop_num) for t in pop_sample]
        out += pop_sample

    out.sort(reverse=descending)
    return out

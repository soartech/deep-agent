from math import pi
from math import cos
from math import floor


def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max, lrate_min):
    """
    Cosine annealing learning rate scheduler
    Reference: https://paperswithcode.com/method/cosine-annealing

    Parameters
    ----------
    epoch: float/int
        Current training epoch
    n_epochs: float/int
        Total number of epochs
    n_cycles: float/int
        Number of cosine cycles to run through (Note: longer cycle time are recommended)
    lrate_max: float
        Max learning rate
    lrate_min: float
        Minimum learning rate

    Returns
    -------
    float
        Current learning rate
    """

    epochs_per_cycle = floor(n_epochs / n_cycles)
    cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
    return lrate_min + ((lrate_max - lrate_min) / 2 * (cos(cos_inner) + 1))


def cosine_annealing_w_attenuation(epoch, n_epochs, n_cycles, lrate_max, lrate_min, k):
    """
    Cosine annealing learning rate scheduler with attenuation
    The max learning rate will decay exponentially over time.

    Parameters
    ----------
    epoch: float/int
        Current training epoch
    n_epochs: float/int
        Total number of epochs
    n_cycles: float/int
        Number of cosine cycles to run through (Note: longer cycle time are recommended)
    lrate_max: float
        Max learning rate
    lrate_min: float
        Minimum learning rate
    k: float
        Exponential decay rate for attenuation, needs to be negative (of the form 10^(k * curr_epoch_num))

    Returns
    -------
    float
        Current learning rate
    """

    if k >= 0:
        raise ValueError(f"Parameter k needs to be less than 0! k: {k}")

    lr = cosine_annealing(epoch, n_epochs, n_cycles, lrate_max, lrate_min)
    return lrate_min + ((lr - lrate_min) * pow(10, -k) * epoch)
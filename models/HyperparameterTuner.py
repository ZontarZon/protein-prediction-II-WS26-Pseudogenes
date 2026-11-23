from math import exp, isfinite, log
from random import Random


def sample(rng, bounds, default):
    low, high = bounds
    if type(default) is int:
        if type(low) is not int or type(high) is not int or low > high:
            raise ValueError("Integer search ranges require ordered integer bounds.")
        return rng.randint(low, high)
    if not (isfinite(low) and isfinite(high) and 0 < low <= high):
        raise ValueError("Log-scale search ranges require ordered, positive finite bounds.")
    return exp(rng.uniform(log(low), log(high))) if low != high else float(low)


def tune(hparams, search_ranges, score, trial_count):
    """Sample integers uniformly and continuous values on a log scale."""
    if trial_count < 1:
        raise ValueError("Hyperparameter tuning requires at least one trial.")
    rng = Random(hparams["seed"])
    trials = []
    best_settings = None
    best_loss = float("inf")
    for _ in range(trial_count):
        overrides = {
            key: sample(rng, bounds, hparams[key])
            for key, bounds in search_ranges.items()
        }
        settings = {**hparams, **overrides}
        loss = score(settings)
        trials.append({"hparams": overrides, "val_loss": loss if isfinite(loss) else None})
        if isfinite(loss) and loss < best_loss:
            best_settings, best_loss = settings, loss
    if best_settings is None:
        raise ValueError("Hyperparameter tuning found no finite validation loss.")
    return best_settings, trials

"""Original free-space and missing-space evaluation metrics."""
import numpy as np

def metrics(prediction, target, valid, observed):
    assert prediction.shape == target.shape and np.isfinite(prediction).all()
    high, low = np.quantile(target[valid], [0.9, 0.1])
    difference = np.abs(prediction - target)
    result = {'thresholds_evaluator_only': [float(high), float(low)]}
    for name, eligible in (('free', valid), ('missing', valid & ~observed)):
        error = difference[eligible]
        result[f'mae_{name}'] = float(error.mean(dtype=np.float64))
        result[f'rmse_{name}'] = float(np.sqrt(np.square(error.astype(float)).mean()))
        for region, truth, guessed in (('hotspot', target >= high, prediction >= high), ('weak', target <= low, prediction <= low)):
            tp = np.count_nonzero(truth & guessed & eligible)
            fp = np.count_nonzero(~truth & guessed & eligible)
            fn = np.count_nonzero(truth & ~guessed & eligible)
            result[f'{region}_f1_{name}'] = float(2 * tp / max(1, 2 * tp + fp + fn))
            regional = difference[truth & eligible]
            result[f'{region}_mae_{name}'] = float(regional.mean(dtype=np.float64)) if regional.size else None
    result['mae_observed'] = float(difference[observed].mean(dtype=np.float64))
    return result

from delta.uncertainty.base import UncertaintyMethod
import numpy as np
from scipy import stats
import pandas as pd


class Ensemble(UncertaintyMethod):

    def train(self, X, Y, stg, cfg, ds_val=None):
        N, D = X.shape

        seed_min = self._seed_start * cfg.confidence.max_n_samples
        for i in range(seed_min, seed_min + self._n_samples):
            # resample dataset:
            ns = np.random.choice(np.arange(N, dtype='int32'), size=N, replace=True)
            X_sample = X[ns, :]
            Y_sample = Y[ns]

            # create and train model:
            model = self._model_factory(i)
            model.train(cfg, X_sample, Y_sample, ds_val=ds_val)
            self._model_samples.append(model)

    def evaluate(self, X, Y, alpha, stg, cfg, logger=None, outfname=None):
        assert len(self._model_samples) > 0, 'no trained models yet'

        N, D = X.shape

        # implement evaluation over bootstraps:
        scores = np.array([model.evaluate(X, Y, stg, log=logger) for model in self._model_samples])

        if logger is not None:
            logger.info('bootstrapped evaluations = %s' % ','.join(map(str, scores)))

        if outfname is not None:
            np.savetxt(outfname, scores)

        return stats.norm.interval(alpha, loc=scores.mean(), scale=scores.std(ddof=1))

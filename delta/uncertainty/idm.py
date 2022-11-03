from delta.uncertainty.base import UncertaintyMethod
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt


class IDM(UncertaintyMethod):
    def train(self, X, Y, stg, cfg, ds_val=None):
        N, D = X.shape
        # create and train model f (without psi) and g (with psi):
        model_f = self._model_factory(self._seed_start)
        model_g = self._model_factory(self._seed_start)

        # train model g:
        model_g.train(cfg, X, Y, ds_val=ds_val)

        if cfg.dataset.hyp.verbose.plot_fits:
            plt.figure()
            (Xval, Yval) = ds_val
            plt.scatter(X, Y.flatten(), marker='x')
            plt.plot(Xval, model_g.predict(Xval), color='red')
            plt.savefig('mse_fit.pdf', dpi=300)

        Xval, Yval = ds_val
        model_g.train_regularized(cfg, X, Y, (Xval[0:1,:],Yval[0:1]))
        # train model f without psi:
        model_f.train(cfg, X, Y, ds_val=ds_val)
        self._model_samples = [model_f, model_g]

    def evaluate(self, X, Y, alpha, stg, cfg, logger=None, outfname=None):
        assert len(self._model_samples) > 0, 'no trained models yet'
        assert alpha == 0.95, 'only 95% confidence currently supported'
        lam = cfg.dataset.hyp.train.rome_eps

        N, D = X.shape

        # implement IDM:
        model_f, model_g = self._model_samples[0], self._model_samples[1]

        psi_0 = model_f.evaluate(X, Y, stg, log=logger)
        psi_lam = model_g.evaluate(X, Y, stg, log=logger)

        V = (psi_lam - psi_0) / lam

        # convert variance V to confidence interval for given alpha:
        n_sided = 2  # 2-sided test
        z_crit = 1.96 #-stats.norm.ppf(alpha / n_sided)
        err = z_crit * np.sqrt(V) / np.sqrt(N)
        lb = psi_0 - err
        ub = psi_0 + err

        scores = np.array([lb, psi_0, ub])

        if logger is not None:
            logger.info('psi_lam = %f, psi_0 = %f, V = %f, 95 percent err = %f' % (psi_lam, psi_0, V, err))
            logger.info('idm %f confidence = %s for z-score = %f' % (alpha, ','.join(map(str, scores)), z_crit))

        if outfname is not None:
            np.savetxt(outfname, scores)

        return lb, ub


class IDMPrediction(IDM):
    def evaluate(self, X, Y, alpha, stg, cfg, logger=None, outfname=None):
        assert len(self._model_samples) > 0, 'no trained models yet'
        assert alpha == 0.95, 'only 95% confidence currently supported'
        lam = cfg.dataset.hyp.train.rome_eps

        N, D = X.shape

        # implement IDM:
        model_f, model_g = self._model_samples[0], self._model_samples[1]

        psi_0 = model_f.evaluate(X, Y, stg, log=logger)
        psi_lam = model_g.evaluate(X, Y, stg, log=logger)

        V = (psi_lam - psi_0) / lam

        # convert variance V to confidence interval for given alpha:
        n_sided = 2  # 2-sided test
        z_crit = 1.96 #-stats.norm.ppf(alpha / n_sided)
        err = z_crit * np.sqrt(V) / np.sqrt(N)
        lb = psi_0 - err
        ub = psi_0 + err

        scores = np.array([lb, psi_0, ub])

        if logger is not None:
            logger.info('psi_lam = %f, psi_0 = %f, V = %f, 95 percent err = %f' % (psi_lam, psi_0, V, err))
            logger.info('idm %f confidence = %s for z-score = %f' % (alpha, ','.join(map(str, scores)), z_crit))

        if outfname is not None:
            np.savetxt(outfname, scores)

        return lb, ub



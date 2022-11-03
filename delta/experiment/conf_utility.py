import logging
from box import Box
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from delta.uncertainty import idm

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> None:

    from scipy import stats
    import numpy as np

    # A logger for this file
    from delta.uncertainty import ensemble
    from delta.model import deepclassifier
    from delta.util import data_util

    hyp = cfg.dataset.hyp

    log.info('applying %s uncertainty to dataset %s with start seed %i' % (cfg.confidence.name, cfg.dataset.name, cfg.confidence.seed_start))

    # 1. load data
    dataset = data_util.Dataset(cfg)
    train_ds, val_ds = dataset.get_train_val()

    # 3. create settings based on data:
    X, Y = train_ds
    X_val, Y_val = val_ds
    # take subset of examples in val:
    X_val = X_val[:hyp.train.val_sample_size,:]
    Y_val = Y_val[:hyp.train.val_sample_size]

    stg = Box({'dataset':
                   {'D': X.shape[1], # num inputs
                    'K': dataset._n_classes, # num unique outputs (assumes Y_val contains no other classes)
                    },
               'seed': hyp.train.seed, # seed for model init
               })

    # 4. build model factory:
    def create_model(seed):
        stg_ = stg.copy()
        stg_['seed'] = seed
        return deepclassifier.FeedForwardNet(cfg, stg_)

    # 5. build model(s) using specified uncertainty method
    if cfg.confidence.name == 'ensemble':
        model = ensemble.Ensemble(create_model, cfg.confidence.n_samples, cfg.confidence.seed_start)
    elif cfg.confidence.name == 'idm':
        model = idm.IDM(create_model, cfg.confidence.n_samples, cfg.confidence.seed_start)
    else:
        raise Exception('confidence method unknown: %s' % cfg.confidence.name)

    # 6. train model knowing evaluation
    def psi(m):
        return m.evaluate(X_val, Y_val, stg)

    model.train(X, Y, stg, cfg, ds_val=val_ds)

    # 7. calc confidence intervals over validation set for a given alpha score
    conf_lb, conf_ub = model.evaluate(X_val, Y_val, cfg.eval.predict_alpha, stg,
                                      cfg, logger=log, outfname='psi_samples.csv')

    # 8. store the result
    log.info('validation utility confidence = (%f, %f) ' % (conf_lb, conf_ub))





if __name__ == "__main__":
    run()
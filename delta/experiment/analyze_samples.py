import logging
import hydra
from omegaconf import DictConfig
from matplotlib import pyplot as plt


log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> None:

    from scipy import stats
    import numpy as np

    # A logger for this file
    from delta.uncertainty import ensemble
    from delta.model import deepclassifier
    from delta.util import data_util

    parent_path = cfg.path.ensemble_samples_proto_path + cfg.path_timestamp
    log.info('analyzing parent path %s with n_seeds = %i and seed_start = %i' % (parent_path, cfg.confidence.n_seeds, cfg.confidence.seed_start))

    n_seeds = cfg.confidence.n_seeds
    seed_start = cfg.confidence.seed_start

    u = []
    for i in range(seed_start, seed_start + n_seeds):
        u_i = np.loadtxt('%s/psi_samples.csv' % (parent_path + ('/%i' % i)))
        u += list(u_i.flatten())
    u = np.array(u)
    N = u.shape[0]

    log.info('samples recovered = %s' % ','.join(map(str, u)))
    log.info('n_samples recovered = %i' % (u.shape[0]))
    log.info('mean = %f, std(ddof=N-1) = %f' % (u.mean(), u.std(ddof=N-1)))
    err = 1.96 * np.sqrt(u.var()) #stats.norm.interval(cfg.eval.predict_alpha, loc=u.mean(), scale=u.std(ddof=u.shape[0]-1))
    conf_lb, conf_ub = u.mean() - err, u.mean() + err
    log.info('validation utility mean, confidence = %f, (%f, %f) ' % (u.mean(), conf_lb, conf_ub))
    np.savetxt(parent_path + '/psi_samples.csv', u)
    if cfg.path.name == 'local':
        plt.hist(u, bins=10)
        plt.errorbar([u.mean()], [0.1], xerr=err, color='red')
        plt.scatter([u.mean()], [0.1], marker='x', color='red')
        plt.xlim(0,4)
        plt.show()

if __name__ == "__main__":
    run()
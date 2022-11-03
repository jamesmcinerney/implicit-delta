import logging
import hydra
from omegaconf import DictConfig
from matplotlib import pyplot as plt
import os

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:

    from scipy import stats
    import numpy as np

    # A logger for this file
    from delta.uncertainty import ensemble
    from delta.model import deepclassifier
    from delta.util import data_util

    parent_path = cfg.path.ensemble_samples_proto_path + cfg.path.timestamp
    log.info('analyzing parent path %s' % (parent_path))

    # crawl iterations to discover seeds
    cs = {'idm':[],
          'delta':[],
          'gp':[]}
    for subdir in os.listdir(parent_path):
        for method in cs.keys():
            try:
                subdir_i = int(subdir)
                # load coverage:
                c = np.loadtxt(os.path.join(parent_path, subdir, 'cvg_%s.csv' % method))
                cs[method].append(c)
            except ValueError:
                pass
            except FileNotFoundError:
                pass
    C = dict([(method, np.array(cs[method])) for method in cs.keys()])

    log.info('n samples found for idm = %i, delta = %i, gp = %i' % (len(cs['idm']), len(cs['delta']), len(cs['gp'])))
    query_xs = np.arange(-2,2,0.05)
    plt.figure()
    plt.plot([query_xs.min(), query_xs.max()], [0.95, 0.95], color='gray')
    plt.plot(query_xs, C['idm'].mean(axis=0), color='green', label='IDM')
    plt.plot(query_xs, C['delta'].mean(axis=0), color='blue', label='Delta', ls='--')
    plt.plot(query_xs, C['gp'].mean(axis=0), color='black', label='GP-Matern52', ls=':')
    plt.xlabel('x')
    plt.ylabel('coverage')
    plt.legend()
    plt.xlim(-2,2)
    plt.ylim(-0.1,1.1)
    plt.savefig('coverage.pdf', dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    run()
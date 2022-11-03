import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, cfg):
        dataset_id = cfg.dataset.name
        np.random.seed(cfg.dataset.generate.seed)

        tfds_catalog = ['mnist']
        uci_catalog = ['satellite','vehicle','waveform']
        synthetic_catalog = ['linear', 'quadratic', 'sin']

        def transform_dat(x, norm_scale = 1.0):
            N, D1, D2 = x.shape
            return np.reshape(x, (N, D1 * D2)) / norm_scale

        self._train_test = None

        if dataset_id in tfds_catalog:
            td = lambda x: transform_dat(x, norm_scale=255)
            (X,Y), (Xval, Yval) = tf.keras.datasets.mnist.load_data() #tfds.load(dataset_id, split=['train','test'], shuffle_files=True, as_supervised=True)
            self._train_test = ((td(X), Y), (td(Xval), Yval))
            self._n_classes = len(np.unique(Y))

        elif dataset_id in synthetic_catalog:
            def td(x):
                N_ = x.shape[0]
                return np.hstack((np.ones((N_,1)), x[:, np.newaxis]))
                #return x[:,np.newaxis]

            psi_stg = cfg.dataset.hyp.psi
            gen = cfg.dataset.generate
            Xval = np.arange(psi_stg.x_min, psi_stg.x_max, psi_stg.x_step)
            if dataset_id == 'sin':
                self._true_fn = lambda x: -np.sin(3*x-0.3)
                X = np.hstack([np.arange(-1.5,-0.7,0.01), np.arange(0.35,1.15,0.01)])
            elif dataset_id == 'quadratic':
                self._true_fn = lambda x: 0.1*x**2 - 0.5*x + 5
                X = np.random.randn(gen.N)
            elif dataset_id == 'linear':
                # set up model mismatch! true generating function is not linear
                self._true_fn = lambda x: 0.1*x**2 - 0.5*x + 5
                X = np.random.randn(gen.N)
            Y = self._true_fn(X) + gen.obs_noise_std*np.random.randn(X.shape[0])
            Yval = self._true_fn(Xval) + gen.obs_noise_std*np.random.randn(Xval.shape[0])
            self._train_test = ((td(X), Y[:,np.newaxis]), (td(Xval), Yval[:,np.newaxis]))
            self._n_classes = 1  # num outputs

        elif dataset_id in uci_catalog:
            if dataset_id == 'satellite':
                dat_train = pd.read_csv(cfg.path.data_path + '/%s/%s_train.data' % (dataset_id, dataset_id),
                                        delimiter=' ', header=None)
                dat_test = pd.read_csv(cfg.path.data_path + '/%s/%s_test.data' % (dataset_id, dataset_id),
                                       delimiter=' ', header=None)
                dat = pd.concat([dat_train, dat_test], axis=0, ignore_index=True)
                # pull out X, Y for train and val
                X = dat.iloc[:, :-1].values.astype('float32')
                Y = pd.get_dummies(dat.iloc[:, -1]).values.argmax(axis=1)
            elif dataset_id == 'vehicle':
                dat = pd.read_csv(cfg.path.data_path + '/%s/%s.dat' % (dataset_id, dataset_id))
                #dat_list = [pd.read_csv(cfg.path.data_path + '/%s/xa%s.dat' % (dataset_id, split_id),
                #                            delimiter='\s*', header=None, skipinitialspace=True) for split_id in
                #                            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']]
                #dat = pd.concat(dat_list, axis=0, ignore_index=True)
                #dat.to_csv('/Users/jmcinerney/Downloads/vehicle_all.csv')
                # pull out X, Y for train and val
                X = dat.iloc[:, 1:-1].values.astype('float32')
                Y = pd.get_dummies(dat.iloc[:, -1]).values.argmax(axis=1)
            elif dataset_id == 'waveform':
                dat = pd.read_csv(cfg.path.data_path + '/%s/%s.data' % (dataset_id, dataset_id), header=None)
                # pull out X, Y for train and val
                X = dat.iloc[:, :-1].values
                Y = pd.get_dummies(dat.iloc[:, -1]).values.argmax(axis=1)

            else:
                raise Exception('dataset not recognized', dataset_id)

            # renormalize inputs:
            X -= X.mean(axis=0)
            X /= (cfg.hyp.constants.epsilon + X.std(axis=0))
            N,D = X.shape
            ns = np.random.binomial(1, p=cfg.dataset.val_fraction*np.ones(N)).astype('bool')
            shuffle_ns = np.random.choice(np.arange(N,dtype='int32'), N, replace=False)
            train_ns = shuffle_ns[~ns]
            val_ns = shuffle_ns[ns]
            Xtrain = X[train_ns,:]
            Ytrain = Y[train_ns]
            Xval = X[val_ns,:]
            Yval = Y[val_ns]
            self._train_test = ((Xtrain,Ytrain),(Xval,Yval))
            self._n_classes = len(np.unique(Y))

        else:
            raise Exception('dataset not recognized', dataset_id)

    def get_train_val(self):
        return self._train_test

    def true_fn(self, x):
        # only applies to synthetic datasets
        return self._true_fn(x)
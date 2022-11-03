import numpy as np
import tensorflow.compat.v1 as tf
from aws import s3
from tensorflow.core.protobuf import saved_model_pb2
import pandas as pd
import itertools as itt
from matplotlib import pyplot as plt
import scipy
from scipy import stats
import logging
import hydra
from omegaconf import DictConfig
import os
from box import Box


# A logger for this file
log = logging.getLogger(__name__)

tf.compat.v1.disable_eager_execution()


class FeedForwardNet(object):

    def __init__(self, cfg: DictConfig, stg: Box):

        D = stg.dataset.D  # dim inputs (first input always 1)
        K = stg.dataset.K # number of outputs

        self._x = tf.placeholder(tf.float64, [None, D], name="x")
        self._x_query = tf.placeholder(tf.float64, [None, D], name="x_query")
        self._lamb = tf.placeholder(tf.float64, [1], name="lamb")
        self._Ntrain = tf.placeholder(tf.float64, [1], name="Ntrain") # total num of examples in dataset

        hyp = cfg.dataset.hyp
        arch = hyp.architecture
        arch.append(K)
        init_bound = hyp.init.weights.uniform_bound

        if hyp.activation == 'tanh':
            activation = tf.nn.tanh
        else:
            raise Exception('unknown activation function')

        np.random.seed(stg.seed)

        self._W = tf.Variable(initial_value=np.random.uniform(-0.1, 0.1, size=[2 + D * arch[0] + arch[0] * arch[1]]))

        def feed_forward(input, dropout=False):
            w0 = tf.reshape(self._W[:D * arch[0]], (D, arch[0]))
            b0 = self._W[D*arch[0]: D*arch[0]+1]
            w1 = tf.reshape(self._W[D * arch[0] + 1:-1], (arch[0], arch[1]))
            b1 = self._W[-1]

            h1 = activation(b0 + tf.linalg.matmul(input, w0))
            if dropout:
                h1_ = tf.nn.dropout(h1, rate=hyp.train.dropout_rate)
            else:
                h1_ = h1
            y = b1 + tf.linalg.matmul(h1_, w1)
            return y

        self._y = feed_forward(self._x, dropout=True)
        self._y_clean = feed_forward(self._x)

        if hyp.loss == 'cross_entropy':
            self._y_type = tf.int64
        elif hyp.loss == 'mse':
            self._y_type = tf.float64

        self._y_obs_query = tf.placeholder(self._y_type, [None,])
        self._y_query = feed_forward(self._x_query)
        self._y_query_dropout = feed_forward(self._x_query, dropout=True)

        if cfg.dataset.hyp.psi.type == 'validation_cross_entropy':
            self._psi = tf.losses.sparse_softmax_cross_entropy(self._y_obs_query, self._y_query)
            self._psi_dropout = tf.losses.sparse_softmax_cross_entropy(self._y_obs_query, self._y_query_dropout)
        elif cfg.dataset.hyp.psi.type == 'prediction':
            self._psi = self._y_query
            self._psi_dropout = self._y_query_dropout
        else:
            raise Exception('psi_type not supported')

        self.create_loss(cfg, K)
        self._psi_history = []
        self._obs_var_fit = 1.0  # scaling to correct for not using log(Normal(x)) in objective

    def create_loss(self, cfg: DictConfig, K):
        hyp = cfg.dataset.hyp

        # now set up the loss:
        self._y_obs = tf.placeholder(self._y_type, [None,])

        if hyp.loss == 'mse':
            self._loss = tf.math.reduce_sum(0.5 * (self._y_obs - self._y) ** 2)
            self._loss_clean = tf.math.reduce_sum(0.5 * (self._y_obs - self._y_clean) ** 2)
        elif hyp.loss == 'cross_entropy':
            self._loss = tf.losses.sparse_softmax_cross_entropy(self._y_obs, self._y)
            self._loss_clean = tf.losses.sparse_softmax_cross_entropy(self._y_obs, self._y_clean)

        self._loss_rome = self._loss - (self._lamb * self._psi_dropout) / self._Ntrain
        self._loss_rome_clean = self._loss_clean - self._lamb * self._psi  / self._Ntrain  # without dropout

        optimizer = hyp.optimizer.type
        lr = hyp.optimizer.lr
        if optimizer == 'gradient_descent':
            loss_optimizer = tf.train.GradientDescentOptimizer(lr)
            loss_rome_optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == 'rms_prop':
            loss_optimizer = tf.train.RMSPropOptimizer(lr)
            loss_rome_optimizer = tf.train.RMSPropOptimizer(lr)
        elif optimizer == 'adagrad':
            loss_optimizer = tf.train.AdagradOptimizer(lr)
            loss_rome_optimizer = tf.train.AdagradOptimizer(lr)
        elif optimizer == 'adam':
            loss_optimizer = tf.train.AdamOptimizer(lr)
            loss_rome_optimizer = tf.train.AdamOptimizer(lr)

        self._train_step = loss_optimizer.minimize(self._loss)
        self._train_step_rome = loss_rome_optimizer.minimize(self._loss_rome)

    def train(self, cfg: DictConfig, X, Y, ds_val=None):
        hyp = cfg.dataset.hyp

        if ds_val is not None:
            Xval, Yval = ds_val

        N = X.shape[0]

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        np.random.seed(hyp.train.seed)

        i = 0
        train_loss_prev = 10
        train_loss = np.inf
        all_ns = np.arange(N, dtype='int32')
        fix_train_ns = np.random.choice(np.arange(N, dtype='int32'), size=hyp.train.train_stopping_size) \
            if hyp.train.batch_size < N else all_ns
        loss_history = Box({'train':[], 'val':[]})
        converge_count = 0
        while i < hyp.train.min_n_iter or (i < hyp.train.max_n_iter and converge_count < hyp.train.max_converge_count):
            # step through examples:
            #start_i = i % N
            #ns = np.arange(start_i, min(N, start_i + hyp.train.batch_size), dtype='int32')
            #if len(ns)==0:
            #    continue
            ns = np.random.choice(all_ns, size=hyp.train.batch_size) if hyp.train.batch_size < N else all_ns

            if (i % hyp.verbose.train_print_every) == 0:
                train_loss_prev = train_loss
                train_loss = sess.run(self._loss_clean, feed_dict={self._x: X[fix_train_ns, :],
                                                     self._y_obs: Y[fix_train_ns]})
                psi = sess.run(self._psi, feed_dict={self._x_query: Xval,
                                                     self._y_obs_query: Yval})
                loss_history.train.append(train_loss)
                loss_history.val.append(psi)
                log.info('iteration %i, fix_train_loss = %f, psi_mean = %f, converge_count = %i' % (i, train_loss, psi.mean(), converge_count))
                if np.abs(train_loss - train_loss_prev) >= hyp.train.threshold_converge:
                    converge_count = max(0, converge_count-1)
                else:
                    converge_count += 1
            sess.run(self._train_step, feed_dict={self._x: X[ns, :],
                                            self._y_obs: Y[ns]})

            i += 1
        for j in range(hyp.train.n_samples_psi):
            # randomly sample examples:
            ns = np.random.choice(all_ns, size=hyp.train.batch_size) if hyp.train.batch_size < N else all_ns
            sess.run(self._train_step, feed_dict={self._x: X[ns, :],
                                                  self._y_obs: Y[ns]})
            self._psi_history.append(sess.run(self._psi, feed_dict={self._x_query: Xval,
                                                                    self._y_obs_query: Yval}))

        log.info('done training.')

        plt.figure()
        plt.plot(hyp.verbose.train_print_every * np.arange(len(loss_history.train)), loss_history.train)
        plt.title('training loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        #plt.savefig(os.path.join(cfg.path.plots, 'training_loss.pdf'), dpi=300)
        plt.savefig('training_loss.pdf', dpi=300)

        if hyp.loss == 'mse':
            self._obs_var_fit = sess.run(tf.reduce_sum((self._y - Y)**2)/(X.shape[0]-1),
                                         feed_dict={self._x: X})

        #log.info('log path = %s' % os.path.join(hydra.run.dir, hydra.sweep.subdir))
        self._sess = sess
        self._fix_train_ns = fix_train_ns

    def train_regularized(self, cfg: DictConfig, X, Y, ds_val):
        hyp = cfg.dataset.hyp

        Xval, Yval = ds_val

        N = X.shape[0]
        all_ns = np.arange(N, dtype='int32')
        fix_train_ns = np.random.choice(np.arange(N, dtype='int32'), size=hyp.train.train_stopping_size) \
            if hyp.train.batch_size < N else all_ns

        i = 0
        lamb = cfg.dataset.hyp.train.rome_eps # scale factor for finite differences
        train_loss_prev = 10
        train_loss = np.inf
        fix_train_ns = self._fix_train_ns
        loss_history = Box({'train':[], 'val':[]})
        converge_count = 0
        while i < hyp.train.max_n_iter and converge_count < hyp.train.max_converge_count:
            ns = np.random.choice(all_ns, size=hyp.train.batch_size) if hyp.train.batch_size < N else all_ns
            if (i % hyp.verbose.train_print_every) == 0:
                train_loss_prev = train_loss
                train_loss = self._sess.run(self._loss_rome_clean, feed_dict={self._x: X[fix_train_ns, :], self._y_obs: Y[fix_train_ns],
                                                             self._x_query: Xval, self._y_obs_query: Yval,
                                                             self._lamb: [lamb], self._Ntrain: [N]})
                psi = self._sess.run(self._psi, feed_dict={self._x_query: Xval,
                                                     self._y_obs_query: Yval})
                loss_history.train.append(train_loss)
                loss_history.val.append(psi)
                log.info('iteration %i, fix_train_loss = %f, psi_mean = %f, converge_count = %i' % (i, train_loss, psi.mean(), converge_count))
                if np.abs(train_loss - train_loss_prev) >= hyp.train.threshold_converge:
                    converge_count = max(0, converge_count-1)
                else:
                    converge_count += 1
            self._sess.run(self._train_step_rome, feed_dict={self._x: X[ns, :], self._y_obs: Y[ns],
                                                             self._x_query: Xval, self._y_obs_query: Yval,
                                                             self._lamb: [lamb], self._Ntrain: [N]})

            i += 1

        self._psi_history = [] # reset psi history
        for j in range(hyp.train.n_samples_psi):
            # randomly sample examples:
            ns = np.random.choice(all_ns, size=hyp.train.batch_size) if hyp.train.batch_size < N else all_ns
            self._sess.run(self._train_step_rome, feed_dict={self._x: X[ns, :], self._y_obs: Y[ns],
                                                             self._x_query: Xval, self._y_obs_query: Yval,
                                                             self._lamb: [lamb], self._Ntrain: [N]})
            self._psi_history.append(self._sess.run(self._psi, feed_dict={self._x_query: Xval,
                                                                    self._y_obs_query: Yval}))

        log.info('done training.')

        #plt.figure()
        #plt.plot(hyp.verbose.train_print_every * np.arange(len(loss_history.train)), loss_history.train)
        #plt.title('training loss')
        #plt.xlabel('iteration')
        #plt.ylabel('loss')
        #plt.savefig('training_loss.pdf', dpi=300)


    def evaluate(self, X, Y, stg, log=None):

        # # calculate predictions:
        # y_pred = self._sess.run([self._y], feed_dict={self._x: X})
        # y_pred_d = y_pred.argmax(axis=1)
        #
        # # evaluate with utility matrix:
        # N, D = X.shape
        # u = np.zeros(N)
        # for i in range(N):
        #     u[i] = stg.utility_matrix[y_pred_d[i], Y[i]]

        u = np.array(self._psi_history)
        if log is not None:
            log.info('utility stats: mean = %f, std = %f' % (u.mean(), u.std()))
            log.info('utility = %s' % ','.join(map(str, u)))

        return u.mean(axis=0)

    def predict(self, X):
        return self._sess.run(self._y_clean, feed_dict={self._x: X})
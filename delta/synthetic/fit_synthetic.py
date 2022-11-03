import logging
import hydra
from box import Box
from omegaconf import DictConfig
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import itertools as itt

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    from delta.util import plotting, data_util, matrix
    import tensorflow.compat.v1 as tf
    import gpflow

    hyp = cfg.dataset.hyp

    # 1. load data
    dataset = data_util.Dataset(cfg)
    train_ds, val_ds = dataset.get_train_val()

    # 2. create settings based on data:
    X, Y = train_ds
    X_val, Y_val = val_ds
    query_xs = np.arange(hyp.psi.x_min, hyp.psi.x_max, hyp.psi.x_step)
    Nquery = query_xs.shape[0]


    stg = Box({'dataset':
                   {'N': X.shape[0], # num examples
                    'D': X.shape[1], # num inputs
                    'K': dataset._n_classes, # num unique outputs (assumes Y_val contains no other classes)
                    },
               'seed': hyp.seed, # seed for model init
               })


    def run_gp(X, Y, query_xs):
        # m = gpflow.models.GPR(data=(X,Y), kernel=gpflow.kernels.ArcCosine())
        m = gpflow.models.GPR(data=(X, Y), kernel=gpflow.kernels.Matern52())

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
        gpflow.utilities.print_summary(m)

        gp_mn, gp_var = m.predict_f(query_xs, full_cov=False)
        _, gp_cov_ = m.predict_f(query_xs, full_cov=True)
        gp_cov = gp_cov_[0, :, :]

        return gp_mn, gp_var, gp_cov

    gp_mn, gp_var, gp_cov = run_gp(X[:,1:2], Y, X_val[:,1:2])
    plotting.plot_cov(gp_cov, fpath='gpmat52_%s_cov.pdf' % cfg.dataset.name)
    plotting.plot_fn_eb(query_xs, gp_mn, np.sqrt(gp_var), data=(X[:, 1], Y),
                        ylims=(hyp.y_min_plt, hyp.y_max_plt),
                        fpath='gpmat52_%s_eb.pdf' % cfg.dataset.name)


    # 3. setup model
    tf.compat.v1.disable_eager_execution()

    D = stg.dataset.D

    x = tf.placeholder(tf.float64, [None, D], name="x")
    x_query = tf.placeholder(tf.float64, [None, D], name="x_query")
    lamb = tf.placeholder(tf.float64, [1], name="lamb")

    arch = hyp.arch

    W = tf.Variable(initial_value=np.random.uniform(-0.1, 0.1, size=[1 + D * arch[0] + arch[0] * arch[1]]))

    wb0 = tf.reshape(W[:D * arch[0]], (D, arch[0]))
    w1 = tf.reshape(W[D * arch[0]:-1], (arch[0], arch[1]))
    b1 = W[-1]

    if hyp.activation == 'tanh':
        activation = tf.nn.tanh
    else:
        raise Exception('unsupported activation')

    h1 = activation(tf.linalg.matmul(x, wb0))
    y = b1 + tf.linalg.matmul(h1, w1)

    h1_query = activation(tf.linalg.matmul(x_query, wb0))
    y_query = b1 + tf.linalg.matmul(h1_query, w1)

    # 4. set up loss
    y_ = tf.placeholder(tf.float64, [None, 1])
    mse = tf.math.reduce_sum(0.5 * (y_ - y) ** 2)
    mse_rome = mse + lamb * tf.reduce_mean(y_query)
    mse_optimizer = tf.train.GradientDescentOptimizer(hyp.mse_lr)
    train_step = mse_optimizer.minimize(mse)
    rome_optimizer = tf.train.GradientDescentOptimizer(hyp.idm_lr)
    train_step_rome = rome_optimizer.minimize(mse_rome)
    reset_optimizer_op = tf.variables_initializer(rome_optimizer.variables())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    # 5. train model

    i = 0
    train_mse_prev = 10
    train_mse = 0
    window_w = []
    train_mse_hist = []
    while i < hyp.n_iter and np.abs(train_mse-train_mse_prev) >= hyp.threshold_converge:
        # full dataset updates:
        ns = np.arange(stg.dataset.N, dtype='int32')
        if (i % cfg.verbose.training.print_every) == 0:
            train_mse_prev = train_mse
            train_mse = sess.run(mse, feed_dict={x: X,
                                            y_: Y})
            log.info('iteration %i, train_mse = %f' % (i, train_mse))
            try:
                float(train_mse)
                pass
            except ValueError:
                log.error('training diverged.')
                return
            train_mse_hist.append(train_mse)
        sess.run(train_step, feed_dict={x: X[ns, :],
                                        y_: Y[ns]})
        # maintain recent history of W's and ensure it does not exceed certain size:
        window_w.append(sess.run(W))
        if len(window_w) > hyp.n_avg_w:
            window_w = window_w[1:]
        i += 1

    #  6. iterate through query points to calculate uncertainty
    query_xs_features = np.hstack((np.ones((Nquery, 1)), query_xs[:, np.newaxis]))

    plt.scatter(X[:, 1], Y, marker='x')

    # average over history:
    pred_ys = np.zeros(Nquery)
    for i in range(hyp.n_avg_w):
        sess.run(W.assign(window_w[i]))
        pred_ys_0 = sess.run(y, feed_dict={x: query_xs_features}).flatten()
        # plt.plot(query_xs, pred_ys_0, color='green')
        pred_ys += pred_ys_0 / hyp.n_avg_w

    plt.plot(query_xs, pred_ys, color='red')

    plt.xlim(hyp.psi.x_min, hyp.psi.x_max)
    plt.ylim(hyp.y_min_plt, hyp.y_max_plt)

    obs_var_fit = sess.run(tf.reduce_sum((y-Y)**2)/(X.shape[0]-1), feed_dict={x: X})

    log.info('obs_var_fit = %f' % obs_var_fit)

    # remember optimal W:
    opt_W = sess.run(W)

    ir_rome = np.zeros((Nquery + 1, Nquery))
    for n in range(Nquery + 1):
        # do M update steps with new y:
        xs_query_n = query_xs_features[n:n + 1, :]
        if n == Nquery:
            xs_query_target = query_xs_features
            denom_y = pred_ys.mean()
        else:
            xs_query_target = np.vstack((query_xs_features[:n, :], query_xs_features[n + 1:, :]))
            denom_y = pred_ys[n]

        i = 0
        # reset parameters to mse optimized:
        # sess.run(reset_optimizer_op)
        sess.run(W.assign(opt_W))

        # calibrate epsilon:
        rome_eps = 0.01 * train_mse
        log.info('rome_eps = %f' % rome_eps)

        n_refine_updates = hyp.n_iter_update if n == 0 else hyp.n_surf_steps
        pred_y_lamb = np.zeros(Nquery)  # maintain average prediction

        train_mse_0 = np.inf
        train_mse_prev = 0
        samp = 0
        pred_y_lamb_hist = []

        while (i < n_refine_updates):  # or (np.abs(train_mse_prev - train_mse_0) > 0.0001): # and np.abs(train_mse-train_mse_prev) >= hyp['threshold_converge_update']:
            # while (i < n_refine_updates):
            # randomly sample examples:
            # ns = np.random.choice(np.arange(N,dtype='int32'), size=hyp['batch_size'])
            ns = np.arange(X.shape[0], dtype='int32')
            train_mse_prev = train_mse_0
            train_mse_0 = sess.run(mse_rome, feed_dict={x: X,
                                                        y_: Y,
                                                        x_query: xs_query_n,  # xs_query_target,
                                                        lamb: [rome_eps]
                                                        })[0]
            if (i % 500) == 0:
                print('iteration %i' % i, train_mse_0)
            sess.run(train_step_rome, feed_dict={x: X[ns, :],
                                                 y_: Y[ns, :],
                                                 x_query: xs_query_n,  # xs_query_target,
                                                 lamb: [rome_eps]
                                                 })
            if (n_refine_updates - i) <= hyp.n_avg_w:
                samp += 1
                if (samp % 100) == 0:
                    print('accumulating sample', samp)
                pred_y_lamb += sess.run(y, feed_dict={x: query_xs_features}).flatten() / hyp.n_avg_w
            # pred_y_lamb_0 = sess.run(y, feed_dict={x: query_xs_features}).flatten()
            # pred_y_lamb_hist.append(pred_y_lamb_0)
            # if len(pred_y_lamb_hist) > hyp['n_avg_w']:
            #    pred_y_lamb_hist = pred_y_lamb_hist[1:]
            # pred_y_lamb = np.array(pred_y_lamb_hist).mean(axis=0)
            i += 1
        # take account of batch size:
        effective_lambda = rome_eps
        ir_rome[n, :] = (pred_y_lamb - pred_ys) / effective_lambda

    cov_rome_ = -ir_rome[:-1, :].T
    try:
        cov_rome = matrix.ensure_psd(cov_rome_, ignore=0)
    except Exception:
        cov_rome = cov_rome_
    sd_rome = np.sqrt(np.diag(cov_rome))

    plotting.plot_fn_eb(query_xs, pred_ys, sd_rome * np.sqrt(obs_var_fit),
                     ylims=(hyp.y_min_plt, hyp.y_max_plt),
                     data=(X[:, 1], Y), fpath='idm_%s_eb.pdf' % cfg.dataset.name)

    # FIM:
    F = X.T @ X
    invF = np.linalg.inv(F)

    # predictive variance for query points:
    pred_var_full = query_xs_features @ invF @ query_xs_features.T
    pred_sd = np.sqrt(np.diag(pred_var_full))

    try:
        plotting.plot_cov(cov_rome * obs_var_fit, fpath='idm_%s_cov.pdf' % cfg.dataset.name)
    except Exception:
        log.error('error plotting covariance')

    # now calculate true Cov using hessian:
    F = sess.run(tf.hessians(mse, W), feed_dict={x: X, y_: Y})[0]

    try:
        # find min value of conditioning weight to make F PSD:
        cond_weight = 0
        cond_incr = 0.1
        while not (matrix.is_pos_def(F + cond_weight * np.eye(F.shape[0]))):
            cond_weight += cond_incr
        print('opt cond_weight', cond_weight)
    except Exception:
        pass

    # take inv, ensure non-singular:
    invF = np.linalg.inv(F + cond_weight * np.eye(F.shape[0]))  # 1e-8*np.eye(F.shape[0]))

    def jacobian():
        # compute jacobian of y w.r.t. W within session sess:
        J = np.zeros((Nquery, F.shape[0]))
        for n in range(Nquery):
            J[n, :] = sess.run(tf.gradients(y, W), feed_dict={x: query_xs_features[n:n + 1, :], y_: np.zeros((1, 1))})[
                0]
        return J


    # predictive variance for query points:
    J = jacobian()
    pred_var_full = J @ invF @ J.T
    pred_sd = np.sqrt(np.diag(pred_var_full))

    plotting.plot_cov(pred_var_full * obs_var_fit, fpath='dm_%s_cov.pdf'  % cfg.dataset.name)
    plotting.plot_fn_eb(query_xs, pred_ys, pred_sd * np.sqrt(obs_var_fit), data=(X[:, 1], Y), ylims=(hyp.y_min_plt, hyp.y_max_plt),
                     fpath="dm_%s_eb.pdf"  % cfg.dataset.name)


    # calculate coverage as function of x
    true_ys = dataset.true_fn(query_xs)

    def save_coverage(pred, sd, true, method):
        conf95 = 1.96 * sd
        lb = pred - conf95
        ub = pred + conf95
        coverage_sample = ((true >= lb) * (true <= ub)).astype('float64')
        np.savetxt('cvg_%s.csv' % method, coverage_sample)

    save_coverage(pred_ys, sd_rome*np.sqrt(obs_var_fit), true_ys, 'idm')
    save_coverage(pred_ys, pred_sd * np.sqrt(obs_var_fit), true_ys, 'delta')
    save_coverage(pred_ys, np.sqrt(np.diag(gp_cov)), true_ys, 'gp')

if __name__ == "__main__":
    run()
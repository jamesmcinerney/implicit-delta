import hydra
from omegaconf import DictConfig, OmegaConf
import logging


log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    import tensorflow.compat.v1 as tf
    import tensorflow_datasets as tfds
    # import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os
    # from tensorflow.examples.tutorials.mnist import input_data

    tf.compat.v1.disable_eager_execution()

    weight_kl = cfg.weight_kl
    weight_recon = cfg.weight_recon

    log.info('weight recon = %f' % weight_recon)
    log.info('adding noise = %f @ fraction %f' % (cfg.obs_noise, cfg.obs_noise_frac))

    #mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    #mnist = Box(tfds.load('mnist'))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # flatten 2D images to 1D:
    def flatten_1d(x):
        N, D1, D2 = x.shape
        return np.reshape(x, (N, D1*D2))

    x_train = flatten_1d(x_train) / 255
    x_test = flatten_1d(x_test) / 255

    test_subset_ind = np.random.choice(x_test.shape[0], size=1000, replace=False)

    # add noise to x_train:
    np.random.seed(0)
    N = x_train.shape[0]
    if cfg.obs_noise_frac > 0:
        def add_noise(x):
            N_ = x.shape[0]
            D_ = x.shape[1]
            noise_ns = np.random.choice(N_, size=int(N_*cfg.obs_noise_frac), replace=False)
            Nn = len(noise_ns)
            x[noise_ns,:] = x[noise_ns,:] + np.random.uniform(0, cfg.obs_noise, size=(Nn,D_))
            return np.clip(x, 0, 1)

        x_train = add_noise(x_train)
        x_test = add_noise(x_test)

    mb_size = 64
    z_dim = 100
    X_dim = x_train.shape[1] #mnist.train.images.shape[1]
    h_dim = 128
    c = 0
    lr = 1e-3

    def plot(samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig


    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)


    # =============================== Q(z|X) ======================================

    X = tf.placeholder(tf.float32, shape=[None, X_dim])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
    Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
    Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

    Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
    Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


    def Q(X):
        h = tf.nn.tanh(tf.matmul(X, Q_W1) + Q_b1) #tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
        z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
        z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
        return z_mu, z_logvar


    def sample_z(mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps


    # =============================== P(X|z) ======================================

    P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
    P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


    def P(z):
        h = tf.nn.tanh(tf.matmul(z, P_W1) + P_b1) #tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
        logits = tf.matmul(h, P_W2) + P_b2
        prob = tf.nn.sigmoid(logits)
        return prob, logits


    # =============================== TRAINING ====================================

    z_mu, z_logvar = Q(X)
    z_sample = sample_z(z_mu, z_logvar)
    _, logits = P(z_sample)

    # Sampling from random z
    X_samples, _ = P(z)

    # E[log P(X|z)]
    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
    # VAE loss
    vae_loss = tf.reduce_mean(weight_recon * recon_loss + weight_kl * kl_loss)

    solver = tf.train.AdamOptimizer().minimize(vae_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    N = x_train.shape[0]
    shuffle_ns = np.random.choice(N, size=N, replace=False)

    for it in range(cfg.set_n_itr):
        start_n = it % N
        end_n = min(start_n + mb_size, N)
        ns = shuffle_ns[start_n : end_n]
        X_mb = x_train[ns, :]

        _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

        if it % 1000 == 0:
            rloss_train, = sess.run([tf.reduce_mean(recon_loss)], feed_dict={X: x_train[test_subset_ind, :]})
            rloss, = sess.run([tf.reduce_mean(recon_loss)], feed_dict={X: x_test[test_subset_ind, :]})
            log.info('Iter: {}'.format(it))
            log.info('Loss: {:.4}'.format(loss))
            log.info('Recon. Loss on Train: {:.4}'.format(rloss_train))
            log.info('Recon. Loss on Held-Out Test: {:.4}'.format(rloss))
            log.info('')

            samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)




if __name__ == "__main__":
    run()
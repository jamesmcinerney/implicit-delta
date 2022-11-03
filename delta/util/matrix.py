import numpy as np




def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def ensure_psd(C, ignore=False):
    if ignore:
        print('ignoring')
        return C.copy()
    if is_pos_def(C):
        print('C already psd')
        return C.copy()
    else:
        print('projecting C to psd')
        return get_near_psd(C)


def run_gp(X, Y, query_xs):
    import gpflow
    # m = gpflow.models.GPR(data=(X,Y), kernel=gpflow.kernels.ArcCosine())
    m = gpflow.models.GPR(data=(X, Y), kernel=gpflow.kernels.Matern52())

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
    gpflow.utilities.print_summary(m)

    gp_mn, gp_var = m.predict_f(query_xs[:, np.newaxis], full_cov=False)
    _, gp_cov_ = m.predict_f(query_xs[:, np.newaxis], full_cov=True)
    gp_cov = gp_cov_[0, :, :]

    return gp_mn, gp_var, gp_cov
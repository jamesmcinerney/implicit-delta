


class UncertaintyMethod(object):
    def __init__(self, model_factory, n_samples, seed_start):
        self._model_factory = model_factory
        self._n_samples = n_samples
        self._seed_start = seed_start
        self._model_samples = []

    def train(self, ds):
        pass

    def confidence(self, ds, zscore):
        pass

    def evaluate(ds, conf):
        # evaluate coverage given dataset `ds` and confidence range `conf`
        pass
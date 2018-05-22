from __future__ import division

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from tensorflow import keras

from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addFilter(NullHandler())

K = keras.backend


class LearningRateUpdater(keras.callbacks.Callback):
    """Callback that updates the learning rate of trained model."""

    def __init__(self, initial_lr, lr_update_rate, n_iter_update, verbose=0):
        """Initialize parameters

        Args:
            initial_lr (float): the initial learning late
            lr_update_rate (float): rate by which the learning rate is updated
            n_iter_update (int): number of iterations of update
            verbose (int): verbosity, defaults to 0
        """
        super(LearningRateUpdater, self).__init__()
        self.initial_lr = initial_lr
        self.lr_update_rate = lr_update_rate
        self.n_iter_update = n_iter_update
        self.verbose = verbose

    def on_batch_begin(self, batch, logs=None):
        if (batch + 1) % self.n_iter_update != 0:
            return

        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        factor = ((batch + 1) // self.n_iter_update) * self.lr_update_rate
        lr = self.initial_lr + factor
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')

        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            logger.info('step {}: Changing learning rate '
                        'to {}.'.format(batch + 1, lr))


class ClusteringLayer(keras.layers.Layer):
    """A Keras layer soft clustering. The output is a vector that represents
    the probability of the sample belonging to each cluster.

    Inputs:
        2D tensor with shape: `(n_samples, n_features)`.
    Outputs:
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, alpha=1.0, weights=None, **kwargs):
        """Initialize parameters

        Args:
            n_clusters (int): number of the clusters
            alpha (float): parameter in Student's t-distribution, defaults to
                1.0
            weights (:obj:`list` of array-like): initial weights of centroids,
                list of ndarray with shape `(n_samples, n_features)`.
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'), )
        super(ClusteringLayer, self).__init__(**kwargs)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self._initial_weights = weights
        self.input_spec = [keras.layers.InputSpec(ndim=2)]

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.centroids = self.add_weight(
            'centroids', (self.n_clusters, input_dim),
            initializer='glorot_uniform')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

        self.input_spec = [
            keras.layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        ]
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(
            K.square(K.expand_dims(inputs, axis=1) - self.centroids), axis=2
        ) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {
            'n_clusters': self.n_clusters,
            'alpha': self.alpha,
        }
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddedClustering(object):
    """Deep embedded clustering model that classifies samples into clusters.
    Attributes:
        encoder (Sequential): encoder model by which samples are transformed
            to new features.
        autoencoder (Sequential): stacked autoencoder model by which samples
            are transformed and reconstructed.
        built (bool): if this is false, pretraining or loading weights is
            needed
        losses (list of float): kullback leibler losses in clustering
        accuracy (list of float): accuracies in clustering, that is available
            if true labels (`y` in `fit` method) are given
        labels (numpy array): the cluster labels predicted by `fit` method
    """

    def __init__(self,
                 input_dim,
                 n_clusters,
                 dims=None,
                 learning_rate=0.01,
                 alpha=1.0,
                 tol=0.01,
                 momentum=0.9,
                 batch_size=256,
                 max_iter=20000,
                 update_interval=None,
                 pretrained_weights=None,
                 verbose=1):
        """Initialize parameters
        Args:
            input_dim (int): dimension of the input features
            n_clusters (int): number of clusters
            dims (list of int): list of numbers of units in stacked autoencoder,
                defaults to [500, 2000, 2000, 10]
            learning_rate (float): the learning rate in clustering
            alpha (int): parameter of clustering layer, defaults to 1.0
            tol (float): the tolerance, clustering ends when the rate of label
                changes falls below this value. Defaults to 0.01.
            momentum (float): parameter of SGD optimizer, defaults to 0.9
            batch_size (int): number of the batch size in clustering, defaults
                to 256
            max_iter (int): maximum number of iteration in clustering, defaults
                to 20000
            update_interval (int): the number of iteration that checks label
                changes, defaults to an epoch.
            pretrained_weights (str): path to the file that is generated by
                this model in pretraining
            verbose (int): verbosity, defaults to 1
        """

        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.dims = dims if dims else [500, 500, 2000, 10]
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.tol = tol
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.update_interval = update_interval
        self.pretrained_weights = pretrained_weights
        self.verbose = verbose
        self.built = False

        # Initialize the stacked autoencoder model
        self.encoders, self.decoders = [], []
        n_input = self.input_dim
        for i, units in enumerate(self.dims, 1):
            encoder_activation = 'linear' if i == len(self.dims) else 'relu'
            encoder = keras.layers.Dense(
                units,
                activation=encoder_activation,
                input_shape=(n_input, ),
                kernel_initializer=keras.initializers.RandomNormal(
                    stddev=0.01),
                name='encoder_{}'.format(i))
            self.encoders.append(encoder)

            decoder_activation = 'linear' if i == 1 else 'relu'
            decoder = keras.layers.Dense(
                n_input,
                activation=decoder_activation,
                kernel_initializer=keras.initializers.RandomNormal(
                    stddev=0.01),
                name='decoder_{}'.format(i))
            self.decoders.append(decoder)
            n_input = units

        self.encoder = keras.models.Sequential(self.encoders)
        self.decoders.reverse()
        self.autoencoder = keras.models.Sequential(
            self.encoders + self.decoders)

        if self.pretrained_weights:
            self.autoencoder.compile('sgd', loss='mse')
            self.autoencoder.load_weights(pretrained_weights)

    @property
    def cluster_centers(self):
        if not self.built:
            return None
        return self.model.layers[-1].get_weights()

    def pretrain(self,
                 X,
                 learning_rate=0.1,
                 batch_size=256,
                 momentum=0.9,
                 drop_fraction=0.2,
                 lr_decay_rate=0.1,
                 n_iter_layer_wise=50000,
                 n_iter_fine_tuning=100000,
                 n_iter_decay=20000,
                 save_path=None):
        """ Pretrain stacked autoencoder
        Args:
            X (numpy array): the training data
            learning_rate (float): learning rate in pretraining, defaults to 0.1
            batch_size (int): number of batch size in pretraining defaults to
                256
            momentum (float): momentum of SGD optimizer, defaults to 0.9
            drop_fraction (float): the rate of dropped features in layer-wise
                pretraining, defaults to 0.2
            lr_decay_rate (float): the reduction rate of leaning rate, defaults
                to 0.1
            n_iter_layer_wise (int): number of iteration in layer-wise
                pretraining, defaults to 50000
            n_iter_fine_tuning (int): number of iteration in fine tuning,
                defaults to 100000
            n_iter_decay (int): number of iteration where the leaning rate is
                updated, defaults to 20000
            save_path (str): path to file where pretrain weights saved, defaults
                to `stacked_autoencoder.h5`
        """

        n_iter_per_epoch = X.shape[0] // batch_size
        layer_wise_epochs = max(n_iter_layer_wise // n_iter_per_epoch, 1)
        fine_tuning_epochs = max(n_iter_fine_tuning // n_iter_per_epoch, 1)
        lr_updater = LearningRateUpdater(learning_rate, lr_decay_rate,
                                         n_iter_decay, self.verbose)

        # Layer-wise pretrain
        if self.verbose > 0:
            logger.info('Layer-wise pretraining...')
        current_x = X
        n_input = self.input_dim
        self.decoders.reverse()
        for i, (encoder, decoder) in enumerate(
                zip(self.encoders, self.decoders), 1):
            if self.verbose > 0:
                logger.info('Pretraining autoencoder_{}'.format(i))
            autoencoder = keras.models.Sequential([
                keras.layers.Dropout(
                    drop_fraction,
                    input_shape=(n_input, ),
                    name='encoder_dropout_{}'.format(i)), encoder,
                keras.layers.Dropout(
                    drop_fraction, name='decoder_dropout_{}'.format(i)),
                decoder
            ])
            autoencoder.compile(
                keras.optimizers.SGD(learning_rate, momentum=momentum),
                loss='mse')
            autoencoder.fit(
                current_x,
                current_x,
                batch_size,
                layer_wise_epochs,
                callbacks=[lr_updater],
                verbose=self.verbose)
            encoder_model = keras.models.Sequential([encoder])
            encoder_model.compile(optimizer='sgd', loss='mse')
            current_x = encoder_model.predict(current_x)
            n_input = self.dims[i - 1]

        # Fine tuning
        if self.verbose > 0:
            logger.info('Fine tuning the stacked autoencoder...')
        self.autoencoder.compile(
            keras.optimizers.SGD(learning_rate, momentum=momentum), loss='mse')
        self.autoencoder.fit(
            X, X, batch_size, fine_tuning_epochs, callbacks=[lr_updater])

        save_path = save_path if save_path else 'stacked_autoencoder.h5'
        self.autoencoder.save_weights(save_path)
        self._initialize_model(X)

    def _initialize_model(self, X):
        self.encoder.compile('sgd', loss='mse')
        if self.verbose > 0:
            logger.info('Initialize centroids with k-means...')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.labels = kmeans.fit_predict(self.encoder.predict(X))
        cluster_centers = kmeans.cluster_centers_

        self.model = keras.models.Sequential([
            self.encoder,
            ClusteringLayer(
                self.n_clusters,
                self.alpha,
                weights=[cluster_centers],
                name='clustering_layer')
        ])
        self.model.compile(
            keras.optimizers.SGD(self.learning_rate, self.momentum),
            loss='kld')
        self.built = True

    def fit(self, X, y=None):
        if not self.built:
            self._initialize_model(X)

        if not self.update_interval:
            self.update_interval = X.shape[0] // self.batch_size

        iteration, index = 0, 0
        self.losses, self.accuracy = [], []

        while True:
            iteration += 1

            if iteration > self.max_iter:
                if self.verbose > 0:
                    logger.info('Reached max iteration, clustering ends.')
                return

            if iteration % self.update_interval == 0:
                y_pred = self.predict(X)
                delta_label = (y_pred != self.labels).sum().astype(
                    np.float32) / y_pred.shape[0]
                self.labels = y_pred

                if y is not None:
                    acc = cluster_acc(y, y_pred)[0]
                    self.accuracy.append(acc)
                    if self.verbose > 0:
                        logger.info('iteration: {}, loss: {}, acc: {}'.format(
                            iteration, self.losses[-1], acc))
                else:
                    if self.verbose > 0:
                        logger.info('iteration {}, loss: {}'.format(
                            iteration, self.loss[-1]))

                if delta_label < self.tol:
                    if self.verbose > 0:
                        logger.info('the rate of changed labels walls below '
                                    'the tolerance and clustering ends.')
                    return

            # train on batch
            if (index + 1) * self.batch_size > X.shape[0]:
                q = self.model.predict(X[index * self.batch_size::])
                p = self.target_distribution(q)
                loss = self.model.train_on_batch(X[index * self.batch_size::],
                                                 p)
                index = 0
            else:
                q = self.model.predict(
                    X[index * self.batch_size:(index + 1) * self.batch_size])
                p = self.target_distribution(q)
                loss = self.model.train_on_batch(
                    X[index * self.batch_size:(index + 1) * self.batch_size],
                    p)
                index += 1
            self.losses.append(loss)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def predict(self, x):
        assert self.built
        q = self.model.predict(x)
        return q.argmax(axis=1)

    @staticmethod
    def target_distribution(q):
        weight = q**2 / q.sum(axis=0)
        return (weight.T / weight.sum(axis=1)).T


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w

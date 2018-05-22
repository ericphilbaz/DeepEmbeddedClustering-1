import numpy as np
from tensorflow import keras

from dec import DeepEmbeddedClustering

# logging
from logging import getLogger, basicConfig, DEBUG
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
basicConfig(level=DEBUG, format=log_format)
logger = getLogger(__name__)


def main():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_all = np.concatenate((X_train, X_test), axis=0)
    X = X_all.reshape(-1, 784)
    y = np.concatenate((y_train, y_test), axis=0)

    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    y = y[p]

    dec = DeepEmbeddedClustering(X.shape[-1], n_clusters=10)
    dec.pretrain(X)
    dec.fit(X, y)


if __name__ == '__main__':
    main()

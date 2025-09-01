from collections.abc import Callable
import numpy as np
import pandas as pd
from config import Config 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


UNKNOWN = -1
np.random.seed(42)


#helper function
def _prep_pca(x_train: np.ndarray, n_components: int = 80):
    """
    Returns a fitted pipeline: Standardize --> PCA(whiten) --> L2 normalize.
    """
    pca = PCA(n_components=n_components, whiten=True, svd_solver="full", random_state=42)
    scaler = StandardScaler(with_mean=True, with_std=True)
    pipe = make_pipeline(scaler, pca)
    x_tr = pipe.fit_transform(x_train)
    x_tr = normalize(x_tr)  # L2 so euclidean
    return pipe, x_tr

def _transform(pipe, x):
    z = pipe.transform(x)
    return normalize(z)

def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'spl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.

    SPL_LABEL = 999_999
    y_spl = np.where(y_train == -1, SPL_LABEL, y_train)

    feat_pipe, x_tr = _prep_pca(x_train, n_components=96)

    clf = LogisticRegression(
        solver="lbfgs",
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(x_tr, y_spl)

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: In this nested function, you can use everything you have trained in the outer
        #       function.
        
        z = _transform(feat_pipe, x_test)
        probs = clf.predict_proba(z)
        classes = clf.classes_
        if SPL_LABEL in classes:
            unk_col = int(np.where(classes == SPL_LABEL)[0][0])
            p_unknown = probs[:, unk_col]
            y_score = (1.0 - p_unknown).astype(float)
        else:
        # fallback: use probability margin as "knownness"
            max_prob = probs.max(axis=1)
            second = np.partition(probs, -2, axis=1)[:, -2]
            y_score = (max_prob - second).astype(float)

        pred_idx = probs.argmax(axis=1)
        y_pred = classes[pred_idx]
        y_pred = np.where(y_pred == SPL_LABEL, -1, y_pred).astype(int)
        return y_pred, y_score
    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'mpl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.

    # feature preparation
    feat_pipe, x_tr = _prep_pca(x_train, n_components=80)

    # split into KCs and KUCs
    kc_mask = y_train != -1
    kuc_mask = ~kc_mask

    x_kc = x_tr[kc_mask]
    y_kc = y_train[kc_mask]
    x_kuc = x_tr[kuc_mask]

    # build KC centroids 
    if x_kc.size == 0:
        # degenerate fallback: "everything unknown"
        def mpl_predict_fn(x_test: np.ndarray):
            return -np.ones(len(x_test), dtype=int), np.zeros(len(x_test), dtype=float)
        return mpl_predict_fn

    unique_kc = np.unique(y_kc)
    kc_centroids = []
    kc_labels = []
    for c in unique_kc:
        kc_centroids.append(x_kc[y_kc == c].mean(axis=0))
        kc_labels.append(c)
    kc_centroids = normalize(np.vstack(kc_centroids))
    kc_labels = np.asarray(kc_labels, dtype=int)

    # build two NN indices (euclidean on L2-normalized == cosine)
    nn_kc = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn_kc.fit(kc_centroids)

    nn_kuc = None
    if x_kuc.size > 0:
        nn_kuc = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nn_kuc.fit(x_kuc)

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
         # TODO: In this nested function, you can use everything you have trained in the outer
        z = _transform(feat_pipe, x_test)

    # nearest KC centroid
        d_kc, idx_kc = nn_kc.kneighbors(z, return_distance=True)
        d_kc = d_kc.ravel()
        idx_kc = idx_kc.ravel()
        pred_kc = kc_labels[idx_kc]

    # cosine similarity from L2-normalized euclidean
        sim_kc = 1.0 - 0.5 * np.square(d_kc)

    # nearest KUC (if present), else very small similarity
        if nn_kuc is not None:
            k = min(3, len(x_kuc))                             # top-3
            d_kuc, _ = nn_kuc.kneighbors(z, n_neighbors=k, return_distance=True)
            sim_kuc = 1.0 - 0.5 * np.square(d_kuc)            # (N, k)
            sim_kuc = sim_kuc.mean(axis=1)                     # calculate average
        else:
            sim_kuc = np.full_like(sim_kc, -1e9)
        margin = sim_kc - sim_kuc

        y_pred = pred_kc.astype(int)         # best KC label
        y_score = margin.astype(float)       
        return y_pred, y_score
    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y



def main():
    x_train, y_train = load_challenge_train_data()

    # TODO: implement
    
    spl_predict_fn = spl_training(x_train, y_train)

    # TODO: implement
    mpl_predict_fn = mpl_training(x_train, y_train)

    # TODO: No todo, but this is roughly how we will test your implementation (with real data). So
    #       please make sure that this call (besides the unit tests) does what it is supposed to do.
    x_test = np.random.rand(50, x_train.shape[1])
    y_test = np.random.randint(-1, 5, 50)
    for predict_fn in (spl_predict_fn, mpl_predict_fn):
        y_pred, y_score = predict_fn(x_test)
        print("Dummy acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))


if __name__ == "__main__":
    main()
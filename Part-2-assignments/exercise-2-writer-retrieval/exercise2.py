import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle
from sklearn.decomposition import PCA
import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
import glob


def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    
    parser.add_argument('--from_images', action='store_true',
                    help='use raw images (SIFT+Hellinger) instead of precomputed .pkl.gz')
    parser.add_argument('--suffix_train', default='.png',
                    help='suffix for TRAIN files (e.g., .png, .jpg)')
    parser.add_argument('--suffix_test', default='.jpg',
                    help='suffix for TEST files (e.g., .png, .jpg)')
    parser.add_argument('--multi_vlad', action='store_true',
                    help='Enable multi-VLAD + PCA whitening (task g).')
    parser.add_argument('--multi_runs', type=int, default=5,
                    help='Number of codebooks to build (default: 5).')
    parser.add_argument('--multi_K', type=int, default=32,
                    help='Clusters per codebook (default: 32; keep small for RAM).')
    parser.add_argument('--multi_descs', type=int, default=200_000,
                    help='Random descriptors per codebook build (default: 200k).')
    parser.add_argument('--pca_dim', type=int, default=1000,
                    help='Target PCA dimensionality (default: 1000).')
    parser.add_argument('--no_whiten', action='store_true',
                    help='Disable PCA whitening (by default whitening is on).')
    parser.add_argument('--esvm_on_multi', action='store_true',
                    help='Also run E-SVM on the multi-VLAD (PCA) features.')

    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ocvmb', '.csv',
          '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        file_name = file_name.replace(' ', '_')  

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels


def build_multiple_codebooks(files_train, runs, K, max_desc, from_images):
    """
    Create 'runs' different codebooks (k-means with different seeds / samples).
    Saves each to mus_run{r}.pkl.gz for reuse.
    Returns: list of KxD arrays (mus_list).
    """
    mus_list = []
    for r in range(runs):
        fname = f'mus_run{r}.pkl.gz'
        if os.path.exists(fname):
            with gzip.open(fname, 'rb') as f:
                mus = cPickle.load(f)
        else:
            # get fresh sample of descriptors for current run
            descs = loadRandomDescriptors(files_train, max_desc, from_images=from_images)
            mus = dictionary(descs, n_clusters=K, seed=42 + r)
            with gzip.open(fname, 'wb') as f:
                cPickle.dump(mus, f, -1)
        mus_list.append(mus)
    return mus_list



def multi_vlad_encode(files, mus_list, powernorm, gmp, gamma, from_images):
    """
    Compute VLAD for each codebook in mus_list and concatenate along feature dim.
    Returns: (N x sum_i(K_i*D)) float32
    """
    encs_per_run = []
    for r, mus in enumerate(mus_list):
        print(f'> compute VLAD for run {r+1}/{len(mus_list)} (K={mus.shape[0]})')
        encs_r = vlad(files, mus, powernorm, gmp=gmp, gamma=gamma, from_images=from_images)
        encs_per_run.append(encs_r.astype(np.float32))
    # concat features horizontally
    return np.hstack(encs_per_run).astype(np.float32)


def fit_pca_whiten(X_train, out_dim, whiten=True):
    """
    Fit PCA (optionally with whitening) on TRAIN encs and return (pca, X_train_pca).
    """
    pca = PCA(n_components=min(out_dim, X_train.shape[1], X_train.shape[0]),
              whiten=whiten, svd_solver='auto', random_state=42)
    Xp = pca.fit_transform(X_train)
    # L2 norm after PCA (common practice)
    Xp = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-12)
    return pca, Xp.astype(np.float32)


def transform_pca_whiten(pca, X):
    Xp = pca.transform(X)
    Xp = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-12)
    return Xp.astype(np.float32)



def computeDescs(filename):
    """
    SIFT descriptors at keypoints (all angles set to 0) + Hellinger normalization.
    Returns: (T x 128) float32 array. Returns empty (0 x 128) if the image
    cannot be read or no keypoints are found.
    """
    # Try to read; if it fails, try common alternative extensions with same basename
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        base, _ = os.path.splitext(filename)
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.TIF', '.TIFF'):
            cand = base + ext
            if os.path.exists(cand):
                img = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    filename = cand
                    break
    if img is None:
        print(f"WARNING: cannot read image: {filename}")
        return np.zeros((0, 128), dtype=np.float32)

    # SIFT keypoints and setting all angles to 0
    sift = cv2.SIFT_create()
    kps = sift.detect(img, None)
    if not kps:
        return np.zeros((0, 128), dtype=np.float32)
    
    for kp in kps:
        kp.angle = 0.0

    # Compute descriptors
    kps, desc = sift.compute(img, kps)
    if desc is None or len(desc) == 0:
        return np.zeros((0, 128), dtype=np.float32)

    # Hellinger: L1 normalize then signed sqrt (no L2 afterwards)
    desc = desc / (np.linalg.norm(desc, ord=1, axis=1, keepdims=True) + 1e-12)
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    return desc.astype(np.float32)



def loadRandomDescriptors(files, max_descriptors,from_images=False):
    """
    Roughly load 'max_descriptors' random descriptors from a subset of 'files'.
    - If from_images=True: compute via computeDescs(filename).
    - Else: load precomputed .pkl.gz arrays.
    Robustness:
       Tries alternative extensions if the constructed path doesn't exist.
       Wildcard-matches variants with extra middle tokens (e.g., '7-*-IMG_MAX_xxx').
       Skips unreadable/missing files.
    Returns: (Q x D) float32 array.
    """

    if len(files) == 0:
        raise RuntimeError("No files provided to loadRandomDescriptors().")
    # let's just take 100 files to speed-up the process
    max_files = min(100, len(files))
    indices = np.random.permutation(len(files))[:max_files]
    files = np.asarray(files, dtype=object)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = max(1, int(max_descriptors // max_files))

    descriptors = []
    for i in tqdm(range(len(files))):
        path = files[i]

        # If path doesn't exist, try common extensions with same basename
        if not os.path.exists(path):
            base, _ = os.path.splitext(path)
            dirname = os.path.dirname(path)
            stem = os.path.basename(base)
            found = None

            # Try direct extension changes
            for ext in ('.jpg', '.png', '.jpeg', '.tif', '.tiff',
                        '.JPG', '.PNG', '.TIF', '.TIFF'):
                cand = base + ext
                if os.path.exists(cand):
                    found = cand
                    break

            # Try wildcard with extra middle token, e.g., '7-*-IMG_MAX_10038.jpg'
            if found is None and '-' in stem:
                left, right = stem.split('-', 1)
                for ext in ('.jpg', '.png', '.jpeg', '.tif', '.tiff',
                            '.JPG', '.PNG', '.TIF', '.TIFF'):
                    pattern = os.path.join(dirname, f"{left}-*-{right}{ext}")
                    hits = glob.glob(pattern)
                    if hits:
                        found = hits[0]
                        break

            if found is None:
                print("MISSING (skipping):", path)
                continue
            path = found

        # Load / compute descriptors
        if from_images:
            desc = computeDescs(path)
        else:
            with gzip.open(path, 'rb') as ff:
                desc = cPickle.load(ff, encoding='latin1')

        if desc is None or len(desc) == 0:
            continue

        # Sample a subset from this file
        take = min(len(desc), max_descs_per_file)
        sel = np.random.choice(len(desc), take, replace=False)
        descriptors.append(desc[sel])

    if not descriptors:
        raise RuntimeError(
            "No descriptors were loaded. Check that your --in_train folder matches the labels "
            "and that the file suffixes are correct."
        )

    return np.concatenate(descriptors, axis=0).astype(np.float32)







def dictionary(descriptors, n_clusters, seed=None):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=10000, verbose=1,
        random_state=seed
    )
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_


def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # TODO
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors.astype(np.float32), clusters.astype(np.float32), k=1)
    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )
    # TODO
    for t,m in enumerate(matches):
        assignment[t, m[0].trainIdx] = 1 # set 1 for nearest cluster center for each descriptor

    return assignment


def vlad(files, mus, powernorm, gmp=False, gamma=1.0, from_images=False):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        if from_images:
            desc = computeDescs(f)
        else:
            with gzip.open(f, 'rb') as ff:
                desc = cPickle.load(ff, encoding='latin1')

        if desc is None or desc.shape[0] == 0:
            encodings.append(np.zeros(K * mus.shape[1], dtype=np.float32))
            continue

        a = assignments(desc, mus)
        
        T,D = desc.shape
        f_enc = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = a[:, k].astype(bool)
            if not np.any(mask):
                continue
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select
            # residuals of descriptors assigned to cluster k
            Rk = desc[mask] - mus[k]        # shape: T_k x D

            if gmp:
                # Generalized Max Pooling via ridge regression 
                # Solve: min_w ||Rk @ w - 1||^2 + gamma ||w||^2
                rr = Ridge(alpha=gamma, solver='sparse_cg',
                           fit_intercept=False, max_iter=500)
                y = np.ones(Rk.shape[0], dtype=np.float32)
                rr.fit(Rk, y)
                f_enc[k] = rr.coef_.astype(np.float32)
            else:
                # Standard VLAD sum-pooling 
                f_enc[k] = np.sum(Rk, axis=0)

        f_enc = f_enc.ravel()
        # c) power normalization
        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))
        # l2 normalization
        nrm = np.linalg.norm(f_enc)
        if nrm > 0:
            f_enc /= nrm
        
        encodings.append(f_enc.astype(np.float32))

    return np.vstack(encodings)
   

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """


    # set up labels
    # TODO
    args = [(i, encs_test, encs_train, C) for i in range(len(encs_test))] 

    def loop(arg):
        # compute SVM 
        # and make feature transformation
        # TODO
        i, encs_test, encs_train, C = arg
        test_descriptor = encs_test[i]
        x_ = np.vstack([encs_train, test_descriptor[np.newaxis, :]])  # training data for SVM
        y_ = np.hstack([np.full(len(encs_train), -1), 1])  

        clf = LinearSVC(C=C, class_weight="balanced", max_iter=10000)  
        clf.fit(x_, y_)

        x = normalize(clf.coef_, norm='l2').flatten()  # l2 normalize the weight vector from the trained SVM
        return x # new global descriptor

    # let's do that in parallel: 
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    new_encs = list(map(loop, tqdm(args))) # builds 1 SVM per test sample
    new_encs = np.stack(new_encs) #  combine all new descriptors into 1 (N, D) matrix 3600x6400
    # return new encodings
    return new_encs

def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO
    dists = 1.0 - np.dot(encs, encs.T)
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix_train, args.labels_train)

    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        # TODO
        descriptors = loadRandomDescriptors(files_train, 500_000, from_images=args.from_images)#its 500_000
        print('> loaded {} descriptors:'.format(len(descriptors)))

        # TODO
        # computed codebook — the set of 100 "visual words" (cluster centers)
        mus = dictionary(descriptors, n_clusters=100)#it was 100
        # cluster centers
        print('> compute dictionary')
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

  
    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test,  labels_test  = getFiles(args.in_test,  args.suffix_test,  args.labels_test)

    print('#test: {}'.format(len(files_test)))
    fname = f'enc_test{"_gmp"+str(args.gamma) if args.gmp else ""}.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_test  = vlad(files_test,  mus, args.powernorm, args.gmp, args.gamma, from_images=args.from_images)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
   
    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = f'enc_train{"_gmp"+str(args.gamma) if args.gmp else ""}.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_train = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma, from_images=args.from_images)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation start')
    # TODO
    enc_test_esvm = esvm(enc_test, enc_train, C=args.C)

    
    # compute pairwise distances and evaluate mAP
    print('> evaluate esvm encodings')
    evaluate(enc_test_esvm, labels_test)


    # -------------------- (g) Multi-VLAD + PCA whitening (optional) --------------------
    if args.multi_vlad:
        print('\n===== TASK (g): multi-VLAD + PCA whitening =====')

    # 1) build multiple codebooks (or load cached ones)
        mus_list = build_multiple_codebooks(
            files_train,
            runs=args.multi_runs,
            K=args.multi_K,
            max_desc=args.multi_descs,
            from_images=args.from_images
            )
        # 2) compute multi-VLAD encodings for TRAIN (fit PCA on train only!)
        print('> multi-VLAD: computing encodings for TRAIN')
        enc_train_multi = multi_vlad_encode(
            files_train, mus_list,
            powernorm=args.powernorm, gmp=args.gmp, gamma=args.gamma,
            from_images=args.from_images)

    # 3) compute multi-VLAD encodings for TEST
        print('> multi-VLAD: computing encodings for TEST')
        enc_test_multi = multi_vlad_encode(
            files_test, mus_list,
            powernorm=args.powernorm, gmp=args.gmp, gamma=args.gamma,
            from_images=args.from_images
            )

    # 4) PCA (whiten=True by default unless --no_whiten)
        print('> PCA fit on TRAIN multi-VLAD (with{} whitening)'.format(
            '' if not args.no_whiten else 'out'
            ))
        pca, enc_train_pca = fit_pca_whiten(
            enc_train_multi, out_dim=args.pca_dim, whiten=(not args.no_whiten)
            )
        enc_test_pca = transform_pca_whiten(pca, enc_test_multi)

    # 5) evaluate multi-VLAD + PCA
        print('> evaluate multi-VLAD + PCA')
        evaluate(enc_test_pca, labels_test)

    # 6) optional: E-SVM on PCA-compressed multi-VLAD
        if args.esvm_on_multi:
            print('> E-SVM on multi-VLAD (PCA) encodings')
            enc_test_esvm_multi = esvm(enc_test_pca, enc_train_pca, C=args.C)
            print('> evaluate E-SVM on multi-VLAD (PCA)')
            evaluate(enc_test_esvm_multi, labels_test)



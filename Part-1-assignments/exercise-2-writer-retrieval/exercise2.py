import os
import shlex
import argparse
from tqdm import tqdm
# for python3: read in python2 pickled files
import _pickle as cPickle
import argparse
import gzip, pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap


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
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

#-----------------------PART (a) Codebook generation-----------------
def dictionary(descriptors, n_clusters):  #dictionary clusters = prototype patches
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    print('> compute dictionary with {} clusters'.format(n_clusters))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, verbose=1, random_state=42)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_ # holds coordinates of the cluster centers aka centroids


#-----------------------PART (b) VLAD encoding-----------------
def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors using OpenCV BFMatcher
    # TODO
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors.astype(np.float32), clusters.astype(np.float32), k=1)
    # TODO

    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )
    # TODO
    for t, m in enumerate(matches):
        assignment[t, m[0].trainIdx] = 1 # set 1 for nearest cluster center for each descriptor

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0] # no. of clusters
    encodings = []   # encodings list

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')

        if desc.shape[0] == 0:
            # handle empty descriptors
            encodings.append(np.zeros(K * mus.shape[1], dtype=np.float32))
            continue    

        a = assignments(desc, mus) # (TxK matrix)
        
        T,D = desc.shape
        f_enc = np.zeros((K, D), dtype=np.float32)
        
        for k in range(mus.shape[0]):
            assigned = a[:, k].astype(bool)  # descriptors assigned to cluster k
            if np.any(assigned):
                residuals = desc[assigned] - mus[k]
                f_enc[k] = np.sum(residuals, axis=0)

        # flatten 
        f_enc = f_enc.flatten()

        # -------------------PART (c) VLAD normalization------------------
        if powernorm:
            # TODO
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # l2 normalization
        # TODO
        norm = np.linalg.norm(f_enc)
        if norm > 0:
            f_enc /= norm

        encodings.append(f_enc)

    return np.vstack(encodings) # numpy matrix (N x K*D)

#-----------------------PART (d) Exemplar classification-----------------
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
    # i --> each test descriptor
    args = [(i, encs_test, encs_train, C) for i in range(len(encs_test))] 

    # inner function to compute 1 esvm
    def loop(arg):
        i, encs_test, encs_train, C = arg
        test_descriptor = encs_test[i]
        x = np.vstack([encs_train, test_descriptor[np.newaxis, :]])  # create training data for SVM
        y = np.hstack([np.full(len(encs_train), -1), 1])  

        clf = LinearSVC(C=C, class_weight="balanced", max_iter=10000)  
        clf.fit(x, y)

        x_global = normalize(clf.coef_, norm='l2').flatten()  # ℓ₂ normalize the weight vector from the trained SVM
        return x_global # new global descriptor

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
    # compute cosine distance = 1 - dot product between l2-normalized encodings
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
    files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)

    print('#train: {}'.format(len(files_train)))

    if not os.path.exists('mus.pkl.gz'):
        # TODO
        descriptors = loadRandomDescriptors(files_train, 500_000)
        print('> loaded {} descriptors:'.format(len(descriptors)))

        # computed codebook — the set of 100 "visual words" (cluster centers)
        mus = dictionary(descriptors, n_clusters=100) 

        # cluster centers
        print('> compute dictionary')
        # TODO
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_test = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)     

    # cross-evaluate test encodings
    print('> evaluate for VLAD')
    evaluate(enc_test, labels_test)        

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_train = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation start')
    enc_test_esvm = esvm(enc_test, enc_train, C=args.C)

    # compute pairwise distances and evaluate mAP
    print('> evaluate esvm encodings')
    evaluate(enc_test_esvm, labels_test)

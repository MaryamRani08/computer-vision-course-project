import os
import pickle
import cv2
import numpy as np
from config import Config 


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def get_embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    
    def __init__(self, num_neighbours=3, max_distance=0.45, min_prob=0.93):

        # 2: We tested the open-set rule with several unknown persons. By tuning the thresholds max_distance = 0.45 and min_prob = 0.93,
        #    the model improves its ability to reject unknowns. Higher min_prob and lower max_distance reduced false acceptances. 
        #    Tuning these values significantly affects open-set performance.

        # 3: Using only one criterion (distance or probability) is not sufficient for reliable open-set identification.
        #    Using both thresholds together gives a more robust decision boundary.
        #    This combination avoids confident misclassifications and catches subtle unknowns.


        # TODO: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # EX 5.2 
    # TODO: Train face identification with a new face with labeled identity.
    def partial_fit(self, face, label):
        clr_embedding = self.facenet.predict(face)
        clr_embedding = clr_embedding / np.linalg.norm(clr_embedding)

        # Convert the face to grayscale
        grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        grayscale_face = cv2.cvtColor(grayscale_face, cv2.COLOR_GRAY2BGR) 
        grayscale_embedding = self.facenet.predict(grayscale_face)

        # L2 normalization.
        grayscale_embedding = grayscale_embedding / np.linalg.norm(grayscale_embedding)

        clr_embedding = clr_embedding.flatten()
        grayscale_embedding = grayscale_embedding.flatten()

        # Adding new embeddings to gallery
        self.embeddings = np.vstack((self.embeddings, clr_embedding, grayscale_embedding))
        self.labels.append(f"{label}_color")  
        self.labels.append(f"{label}_gray") 

    # TODO: Predict the identity for a new face.
    def predict(self, face):
         # build query embedding
        emb_color = self.facenet.predict(face)
        emb_color /= np.linalg.norm(emb_color)

        # convert itno greyscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        emb_gray = self.facenet.predict(gray)
        emb_gray /= np.linalg.norm(emb_gray)

        query = (emb_color + emb_gray) / 2.0
        query /= np.linalg.norm(query)

        # check if empty
        if self.embeddings.shape[0] == 0:
            return "unknown", 0.0, float("inf")

        # calculate distances
        diffs   = self.embeddings - query                      
        numer  = np.einsum("ij,ij->i", diffs, diffs)         
        dist   = numer / 2.0                              

        # calculate k-nearest neighbours 
        k = min(self.num_neighbours, len(dist))
        knn_idx       = np.argsort(dist)[:k]
        knn_labels    = [self.labels[i] for i in knn_idx]

        base_labels   = [lbl.split("_")[0] for lbl in knn_labels]
        predicted_lables     = max(set(base_labels), key=base_labels.count)

        # calculate Posterior Probability 
        k_i           = base_labels.count(predicted_lables)
        prob          = k_i / k

        # calculate min distance to predicted class  
        class_idxs    = [idx for idx, bl in zip(knn_idx, base_labels) if bl == predicted_lables]
        mini_dists    = dist[class_idxs].min()

        # open‑set decision rule
        if mini_dists > self.max_distance or prob < self.min_prob: # Using both distance and posterior gives more robust results
            return "unknown", prob, mini_dists

        return predicted_lables, prob, mini_dists


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=25):
        # TODO: Prepare FaceNet.
        self.facenet = FaceNet()

        # embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        self.cluster_center = np.empty((num_clusters, FaceNet.get_embedding_dimensionality))
        self.cluster_membership = []

        self.max_iter = max_iter

        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )

    # EX 5.3
    # TODO
    def partial_fit(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack((self.embeddings, embedding))
        return embedding

    # TODO
    def fit(self):
        if self.embeddings.shape[0] < self.num_clusters:
           raise ValueError("Need at least k embeddings before clustering.")

        # random initialization of k cluster centers
        rnd_idx = np.random.choice(self.embeddings.shape[0], self.num_clusters, replace=False)
        clus_centers = self.embeddings[rnd_idx]

        for _ in range(self.max_iter):
            # assignment and update
            dists = np.linalg.norm(self.embeddings[:, None] - clus_centers[None, :], axis=2)  # (N, k)
            labels = np.argmin(dists, axis=1)  
            new_centers = np.zeros_like(centers)
            for j in range(self.num_clusters):
                members = self.embeddings[labels == j]
                if len(members) > 0:
                    new_centers[j] = members.mean(axis=0)
                else:
                    # Reinitialize it with a random embedding in case it is empty
                    new_centers[j] = self.embeddings[np.random.randint(self.embeddings.shape[0])]

            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        self.cluster_center = centers
        self.cluster_membership = labels

    # TODO
    def predict(self, face):
        embedding = self.facenet.predict(face).flatten()
        embedding /= np.linalg.norm(embedding)

        distance_vector = np.linalg.norm(self.cluster_center - embedding, axis=1)
        best_cluster = int(np.argmin(distance_vector))
        return best_cluster, distance_vector
import pickle

import numpy as np

from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        # Train the Classifier with all training data 
        self.classifier.fit(self.train_embeddings, self.train_labels)

        pred_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        # Initialize list
        identification_rates = []
        similarity_thresholds = []
        false_alarm_rates = []

        for false_alarm_rate in self.false_alarm_rate_range:
            # Get threshold for current false alarm rate
            sim_thr = self.select_similarity_threshold(similarities, false_alarm_rate)
            similarity_thresholds.append(sim_thr)

            # Apply the open-set rule:
            final_preds = np.where(similarities >= sim_thr, pred_labels, UNKNOWN_LABEL)

            # Calculate the identification rate with the corrected predictions
            identification_rate = self.calc_identification_rate(final_preds)
            identification_rates.append(identification_rate)

            gt = np.asarray(self.test_labels)
            false_accepts = np.sum((gt == UNKNOWN_LABEL) & (final_preds != UNKNOWN_LABEL))
            total_unknowns = np.sum(gt == UNKNOWN_LABEL)
            far = false_accepts / total_unknowns if total_unknowns > 0 else 0.0

        return {
            "false_alarm_rates": false_alarm_rates,
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }


    def select_similarity_threshold(self, similarity, false_alarm_rate):
        sims_unknown = similarity[np.asarray(self.test_labels) == UNKNOWN_LABEL]
        
        # Ensure that no false acceptances occur when there are no unknown faces
        if sims_unknown.size == 0:
            return float("inf")  

        perc = 100.0 * (1.0 - false_alarm_rate)
        threshold = np.percentile(sims_unknown, perc, interpolation="nearest")
        return threshold

    def calc_identification_rate(self, prediction_labels):
        prediction_labels = np.asarray(prediction_labels)
        gt = np.asarray(self.test_labels)

        known_mask   = gt != UNKNOWN_LABEL
        if not known_mask.any():
            return 0.0

        correct = np.sum(prediction_labels[known_mask] == gt[known_mask])
        return correct / np.sum(known_mask)
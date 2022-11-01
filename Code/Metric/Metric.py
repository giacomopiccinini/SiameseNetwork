import numpy as np


class Metric:

    """Class for evaluating results of predictions in classification"""

    def __init__(self, y_truth, y_pred):

        # Store ground truths and predictions
        self.y_truth = y_truth
        self.y_pred = y_pred

    def compute_binary_crossentropy(self):

        """Compute binary cross entropy"""

        self.y_pred = np.clip(self.y_pred, 1e-7, 1 - 1e-7)

        # Compute first term
        term_0 = (1-self.y_truth) * np.log(1-self.y_pred + 1e-7)

        # Compute second term
        term_1 = self.y_truth* np.log(self.y_pred + 1e-7)

        # Compute total cross entropy
        self.binary_crossentropy = -np.mean(term_0+term_1, axis=0)

    def evaluate(self):

        """Evaluate all metrics"""

        # Compute all metrics
        self.compute_binary_crossentropy()

        # Store all metrics in a dictionary
        metrics = {}

        metrics["binary_crossentropy"] = self.binary_crossentropy

        return metrics
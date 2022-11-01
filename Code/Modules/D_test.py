import numpy as np
import pandas as pd
from Code.Metrics.Metric import Metric


def test(Siamese, experiment, test_set):

    # Load best model
    Siamese.load_weights(f"Output/{experiment.get_name()}/Checkpoints/model.hdf5")

    with experiment.test():

        # Get ground truth
        y_truth = np.array([label.label for label in test_set.labels])

        # Predict values
        y_pred = Siamese.predict(test_set).reshape(-1)

        # Instantiate metric object
        metric = Metric(y_truth=y_truth, y_pred=y_pred, regress=True)

        # Evaluate metrics
        metrics = metric.evaluate()

        # Log metrics
        experiment.log_metrics(metrics)

        # End experiment
        experiment.end()

        # Create pandas data frame
        df = pd.DataFrame({"Ground Truth": y_truth, "Prediction": y_pred})

        # Write results
        df.to_excel(
            f"Output/{experiment.get_name()}/predictions.xlsx",
            sheet_name="sheet1",
            index=False,
        )
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    clf = GDA()
    clf.fit(x_train, y_train)

    print('theta in Gaussian Discriminant Analysis: ', clf.theta)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    print('Accuracy in Gaussian Discriminant Analysis: ',
          np.mean(clf.predict(x_eval) == y_eval))

    preds = clf.predict(x_eval)

    with open(pred_path, 'w') as f:
        for pred in preds:
            f.write(str(pred) + '\n')

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        # Initialize theta
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        # Compute phi, mu_0, mu_1, sigma
        y_1 = sum(y == 1)
        phi = y_1 / m
        # NOTE: axis=0 -> rows, axis=1 -> cols
        mu_0 = np.sum(x[y == 0], axis=0) / (m - y_1)
        mu_1 = np.sum(x[y == 1], axis=0) / y_1
        sigma = (
            (x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) +
            (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)
        ) / m

        # Compute theta
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = (
            mu_0.T.dot(sigma_inv).dot(mu_0) - mu_1.T.dot(sigma_inv).dot(mu_1)
        ) / 2
        self.theta[1:] = sigma_inv.dot(mu_1 - mu_0)

        return

        # *** END CODE HERE ***

    def predict(self, x) -> list[int]:
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y_preds: list[int] = []
        for elem in x:
            y = round(util.sigmoid(self.theta.dot(elem)))
            y_preds.append(y)
        return y_preds
        # *** END CODE HERE

import numpy as np

import util
from linear_model import LinearModel


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # Initialize theta based on number of features
        m, n = x.shape
        self.theta = np.zeros(n)

        # Newton's method
        while True:
            # Save old theta to compare later
            old_theta = np.copy(self.theta)

            # Compute Hessian Matri
            # '../assets/logistic_hypothesis_function.png'
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            H = (x.T * h_x * (1 - h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x - y) / m

            # Update theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)

            # End training if norm of theta_k - theta_k-1 < epsilon
            if np.linalg.norm(self.theta - old_theta, ord=1) < self.eps:
                break

    def predict(self, x) -> list[int]:
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        y_preds: list[int] = []
        for elem in x:
            y = round(util.sigmoid(self.theta.dot(elem)))
            y_preds.append(y)
        return y_preds


x_train, y_train = util.load_dataset('data/ds1_train.csv', add_intercept=True)

x_valid, y_valid = util.load_dataset('data/ds1_valid.csv', add_intercept=True)

clf = LogisticRegression(theta_0=0)
clf.fit(x_train, y_train)
preds = clf.predict(x_valid)

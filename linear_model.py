from typing import Optional
import numpy as np


class LinearModel():
    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0: Optional[float] = None, verbose=True) -> None:
        """
        Args:
            step_size (float, optional): Step size for each iteration. Defaults to 0.2.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            eps (_type_, optional): Threshold to determine convergence. Defaults to 1e-5.
            theta_0 (Optional[float], optional): Initial theta value. Defaults to None.
            verbose (bool, optional): Print loss values during model training. Defaults to True.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Run solver to fit liner model using training data.

        Args:
            x (np.ndarray): Training example inputs. Shape(m, n)
            y (np.ndarray): Training example labels. Shape(m,)
        """
        raise NotImplementedError(
            'Subclass of LinearModel must implement fit method.'
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """"""
        """Make a prediction given new inputs x.

      Args:
          x (np.ndarray): Inputs of shape (m, n).
      Returns:
          Outputs of shape (m,).
      """
        raise NotImplementedError(
            'Subclass of LinearModel must implement predict method.'
        )

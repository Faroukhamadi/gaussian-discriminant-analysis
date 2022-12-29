# Gaussian discriminant analysis

This is a simple implementation of the Gaussian Discriminant Analysis algorithm in Python. It is based on the [Wikipedia article](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Gaussian_discriminant_analysis) on the subject.

## Usage

```python
# Create a GDA object
gda = GDA()

# Train the model
gda.fit(X, y)

# Predict the class of a new sample
gda.predict(X_new)
```

Or, Run the [notebook](gaussian_discriminant_analysis_notebook.ipynb) to see the algorithm in action.

## Implementation details

You can find the implementation in the [gaussian_discriminant_analysis.py](gaussian_discriminant_analysis.py) file.

# ML-Project
## Getting Started
The typical link to the dataset is 
https://yann.lecun.com/exdb/mnist/
but its locked by username and password so if we use
```python
from keras.datasets import mnist

# This takes a while because it's a large dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data(path="mnist.npz")

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
```
we can load the dataset as a Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)` as outlined in the documentation
https://keras.io/api/datasets/mnist/

To use this, you must install Keras using
```
pip install keras
python3 -m pip install tensorflow
```
# ML-Project

## Global Variables

This is how global variables work in python
```python
def main():
    global x
    x = "I want x to remain this value"
    print(x)
    myfunc()

def myfunc():
    x = "This doesnt change the global value"
    print(x)

main()
print(x)
```
Outputs:
```
I want x to remain this value
This doesnt change the global value
I want x to remain this value
```

## Getting Started

The dataset can be found on the <a href="http://yann.lecun.com/exdb/mnist/">MNIST homepage</a>. However we can also import the data using keras:

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

We can download the dataset from
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

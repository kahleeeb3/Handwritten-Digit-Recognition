from keras.datasets import mnist

# This takes a while because it's a large dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data(path="mnist.npz")

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
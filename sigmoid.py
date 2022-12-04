# 
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# This takes a while because it's a large dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def fix_y(y):
    numElem = len(y)
    table = np.zeros((numElem, 10))
    for i in range(numElem):
        table[i][y[i]] = 1 
    return table

X_train = X_train.reshape(60000,784)/255
X_test = X_test.reshape(10000,784)/255

y_train = fix_y(y_train)
y_test = fix_y(y_test)

batch = 64
lr = 1e-3
epochs = 50

x = X_train[:batch]
y = y_train[:batch]

loss = []
acc = []

W1 = np.random.randn(784,256)
W2 = np.random.randn(256,128)
W3 = np.random.randn(128,10)

b1 = np.random.randn(256)
b2 = np.random.randn(128)
b3 = np.random.randn(10)

# def ReLU(x):
#     return np.maximum(0,x)

# def dReLU(x):
#     return 1 * (x > 0) 

def softmax(z):
    z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)

#Sigmoid funstion
def sigmoid(x):
    return 1/(np.exp(-x)+1)    

#derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def shuffle():
    idx = [i for i in range(X_train.shape[0])]
    np.random.shuffle(idx)
    return X_train[idx],y_train[idx]

def test(x,y):
   
    # feedforward()
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    z3 = a2.dot(W3) + b3
    a3 = softmax(z3)
    
    error = a3 - y
    
    acc = np.count_nonzero(np.argmax(a3,axis=1) == np.argmax(y,axis=1)) / x.shape[0]
    print("Accuracy:", 100 * acc, "%")


l = 0
acc = 0
shuffle()

for epoch in range(epochs):
    for chunk in range(X_train.shape[0]//batch-1):
        start = chunk*batch
        end = (chunk+1)*batch
        x = X_train[start:end]
        y = y_train[start:end]

        # feedforward()
        z1 = x.dot(W1) + b1
        a1 = sigmoid(z1)

        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)

        z3 = a2.dot(W3) + b3
        a3 = softmax(z3)

        error = a3 - y

        # backprop()
        dcost = (1/batch)*error

        a = np.dot((dcost),W3.T)
        b = d_sigmoid(z2)
        c = d_sigmoid(z1)
        d = np.dot( a * b ,W2.T)

        DW3 = np.dot(dcost.T,a2).T
        DW2 = np.dot(( a  *  b ).T,a1).T
        DW1 = np.dot((d * c ).T,x).T

        db3 = np.sum(dcost,axis = 0)
        db2 = np.sum( a  *  b ,axis = 0)
        db1 = np.sum((d * c ),axis = 0)

        W3 = W3 - lr * DW3
        W2 = W2 - lr * DW2
        W1 = W1 - lr * DW1

        b3 = b3 - lr * db3
        b2 = b2 - lr * db2
        b1 = b1 - lr * db1


        l+=np.mean(error**2)
        acc+= np.count_nonzero(np.argmax(a3,axis=1) == np.argmax(y,axis=1)) / batch
    print(f"finished epoch {epoch+1}")
    
test(X_test,y_test)
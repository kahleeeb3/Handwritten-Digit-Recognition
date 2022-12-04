import numpy as np
from keras.datasets import mnist

def ReLU(x):
    return np.maximum(0,x)

def dReLU(x):
    return 1 * (x > 0) 

def softmax(z):
    z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)

# #Sigmoid funstion
# def sigmoid(x):
#     return 1/(np.exp(-x)+1)    

# #derivative of sigmoid
# def d_sigmoid(x):
#     return (np.exp(-x))/((np.exp(-x)+1)**2)

def shuffle():
    indices = np.arange(60000) # list from 0 -> 60,000
    np.random.shuffle(indices) # shuffle the index values
    return X_train[indices],y_train[indices]

def getAccuracy(x,y):
   
    # feedforward()
    z1 = x.dot(W1) + b1
    x2 = ReLU(z1)
    
    z2 = x2.dot(W2) + b2
    x3 = ReLU(z2)

    z3 = x3.dot(W3) + b3
    x4 = softmax(z3)
    
    predictedVals = np.argmax(x4,axis=1)
    expectedVals = np.argmax(y,axis=1)
    numCorect = np.count_nonzero(predictedVals == expectedVals)

    print(f"Accuracy: {(numCorect)/100}%") # (1/10000)*100 = 1/100


def fix_y(y):
    numElem = len(y)
    table = np.zeros((numElem, 10))
    for i in range(numElem):
        table[i][y[i]] = 1 
    return table

def forwardPropagation():
    pass

def backwardPropagation():
    pass

# This takes a while because it's a large dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,784)/255
X_test = X_test.reshape(10000,784)/255

y_train = fix_y(y_train)
y_test = fix_y(y_test)

batch = 64
lr = 1e-3
epochs = 10

# x1 = X_train[:batch]
# y = y_train[:batch]

# define number of nodes in each layer
L1N = 784 # Constant
L2N = 256
L3N = 128
L4N = 10 # Constant

W1 = np.random.randn(L1N,L2N)
W2 = np.random.randn(L2N,L3N)
W3 = np.random.randn(L3N,L4N)

b1 = np.random.randn(256)
b2 = np.random.randn(128)
b3 = np.random.randn(10)


for epoch in range(epochs):

    shuffle() # shuffle the data arround
    numChunks = (60000//batch)-1 # number of chunks needed

    for chunk in range(numChunks):
        start = chunk*batch
        end = (chunk+1)*batch
        x1 = X_train[start:end]
        y = y_train[start:end]

        # forwardprop()
        z1 = x1.dot(W1) + b1
        x2 = ReLU(z1)

        z2 = x2.dot(W2) + b2
        x3 = ReLU(z2)

        z3 = x3.dot(W3) + b3
        x4 = softmax(z3)

        error = x4 - y

        # backprop()
        cost = (1/batch)*error

        dz1 = dReLU(z1)
        dz2 = dReLU(z2)
        
        EW2 = np.dot((cost),W3.T)*dz2 # Err in W2
        EW1 = np.dot(EW2 ,W2.T)*dz1 # Err in W1

        DW3 = np.dot(cost.T,x3).T
        DW2 = np.dot(EW2.T,x2).T
        DW1 = np.dot(EW1.T,x1).T

        db3 = np.sum(cost,axis = 0)
        db2 = np.sum(EW2,axis = 0)
        db1 = np.sum(EW1,axis = 0)

        W3 = W3 - lr * DW3
        W2 = W2 - lr * DW2
        W1 = W1 - lr * DW1

        b3 = b3 - lr * db3
        b2 = b2 - lr * db2
        b1 = b1 - lr * db1

    print(f"finished epoch {epoch+1}")
    
getAccuracy(X_test,y_test)
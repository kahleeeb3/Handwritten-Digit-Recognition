import numpy as np
from functions import * # import all from functions.py
from keras.datasets import mnist
import time # for calc runtime

def main():

    # I want these values to remain the same globally
    # any changes inside another func are not affected here
    global X_train, y_train, X_test, y_test

    # load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # format the dataset
    X_train = X_train.reshape(60000,784)/255
    X_test = X_test.reshape(10000,784)/255
    y_train = fix_y(y_train)
    y_test = fix_y(y_test)

    # train the algorithm using these optimal hyperparameters
    train(epochs = 10, batch = 64, lr = 1e-3, L2N = 350, L3N = 350)

    # ================================================================
    # this is for collecting data
    """
    # testing learning rate
    for i in [2,3,4,5,6]:
        lr = 1*10**(-(i))
        train(10,64,lr,350,350)

    # testing hidden layer count
    train(10,64,1e-3,100,16) # 100 and 16
    train(10,64,1e-3,50,128) # 50 and 128
    train(10,64,1e-3,128,128) # 128 and 128
    train(10,64,1e-3,128,256) # 128 and 256
    train(10,64,1e-3,256,128) # 256 and 128
    train(10,64,1e-3,256,256) # 256 and 256
    train(10,64,1e-3,300,300) # 300 and 300
    train(10,64,1e-3,350,350) # 350 and 350
    train(10,64,1e-3,400,400) # 400 and 400

    # testing number of epochs
    train(epochs = 50, batch = 64, lr = 1e-3, L2N = 350, L3N = 350)
    # testing batch size
    for i in [200]:
        st = time.time() # get the start time
        train(epochs = 10, batch = i, lr = 1e-3, L2N = 350, L3N = 350)
        et = time.time() # get the end time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
    """
    # ================================================================


def train(epochs,batch,lr,L2N,L3N):

    # Print the hyperparameter values
    print(f"epochs:{epochs}, lr:{lr}, L2N:{L2N}, L3N:{L3N}, BS:{batch}")

    # I want the global val of these params
    # to only be affected in this function
    global W1, W2, W3, b1, b2, b3

    # define number of nodes in each layer
    L1N = 784 # Constant
    L4N = 10 # Constant

    # initialize values
    W1 = np.random.randn(L1N,L2N)
    W2 = np.random.randn(L2N,L3N)
    W3 = np.random.randn(L3N,L4N)

    b1 = np.random.randn(L2N)
    b2 = np.random.randn(L3N)
    b3 = np.random.randn(L4N)

    # test initial accuracy
    accuracy = getAccuracy() # use the weights to get accuracy
    print(accuracy,end =", ")

    for epoch in range(epochs):

        # shuffle the data
        indices = np.arange(60000) # list from 0 -> 60,000
        np.random.shuffle(indices) # shuffle the index values
        X_train[indices],y_train[indices]


        numChunks = (60000//batch)-1 # number of chunks needed

        for chunk in range(numChunks):
            start = chunk*batch
            end = (chunk+1)*batch
            x1 = X_train[start:end]
            y = y_train[start:end]

            # Forward Propagation
            z1 = x1.dot(W1) + b1
            x2 = phi(z1)

            z2 = x2.dot(W2) + b2
            x3 = phi(z2)

            z3 = x3.dot(W3) + b3
            x4 = softmax(z3)

            error = x4 - y

            # Backward Propagation
            cost = (1/batch)*error

            dz1 = dphi(z1)
            dz2 = dphi(z2)
            
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
    
        # print(f"finished epoch {epoch+1}")
        accuracy = getAccuracy() # use the weights to get accuracy
        print(accuracy,end =", ")
    print("\n")

def getAccuracy():

    # Forward Propagation
    z1 = X_test.dot(W1) + b1
    x2 = phi(z1)
    
    z2 = x2.dot(W2) + b2
    x3 = phi(z2)

    z3 = x3.dot(W3) + b3
    x4 = softmax(z3)
    
    predictedVals = np.argmax(x4,axis=1)
    expectedVals = np.argmax(y_test,axis=1)
    numCorect = np.count_nonzero(predictedVals == expectedVals)

    return (numCorect)/100 # (1/10000)*100 = 1/100

def fix_y(y):
    numElem = len(y)
    table = np.zeros((numElem, 10))
    for i in range(numElem):
        table[i][y[i]] = 1 
    return table


# keep these lines for ReLU
def phi (x) : return ReLU (x)
def dphi (x) : return dReLU (x)

# keep these lines for sigmoid
# def phi (x) : return sigmoid (x)
# def dphi (x) : return d_sigmoid (x)


main()
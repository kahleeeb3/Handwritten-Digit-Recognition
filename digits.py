from keras.datasets import mnist
import numpy as np

#Sigmoid function
def sigmoid(x):
    return 1/(np.exp(-x)+1)    

# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# format data
x_train = x_train.reshape(60000,784)/255
x_test = x_test.reshape(10000,784)/255

epochs = 20 # number of epochs
nodes = 128 # number of nodes in the hidden layer
eta = 1e-3 # learning rate

# need to initialize weights and biases to a random value that is
# uniformly distributed over the half-open interval [-1, 1).

# w1 = np.random.uniform(-1,1,size=(28*28,nodes)) # size (784, nodes)
# w2 = np.random.uniform(-1,1,size=(nodes,10)) # size (nodes, 10)

w1 = np.genfromtxt('w1.csv', delimiter=',')
w2 = np.genfromtxt('w2.csv', delimiter=',')

# Run the simulation
samples = len(x_train)
for e in range(epochs):
    for i in range(samples):

        # format y for the sample
        expected = y_train[i] # answer
        y = np.zeros(10)
        y[expected-1] = 1 # expected value

        # find the values for each node
        x1 = x_train[i] # sample 1 attributes (784)
        z1 = np.dot(x1,w1)

        x2 = sigmoid(z1)
        z2 = np.dot(x2,w2)

        x3 = sigmoid(z2)
                
        # update d2
        dk = x3*(1-x3)*(y-x3)
        dw2 = eta*np.dot(x2[:,None],dk[None,:]) # eta* (dk.x2)


        # update d1
        dh = x2*(1-x2)*np.sum(np.dot(w2,dk))
        dw1 = eta*np.dot(x1[:,None],dh[None,:]) # eta* (dh.x1)

        w1 = w1+dw1
        w2 = w2+dw2
        
    print(f"Ran Epoch {e+1}")

# Test the accuracy
samples = len(x_test)
numWrong = 0

for i in range(samples):
    
    # format y for the sample
    expected = y_test[i] # answer
    y = np.zeros(10)
    y[expected-1] = 1 # expected value
    
    # find the values for each node
    x1 = x_test[i] # sample 1 attributes (784)
    z1 = np.dot(x1,w1)

    x2 = sigmoid(z1)
    z2 = np.dot(x2,w2)

    x3 = sigmoid(z2)
    
    predicted = np.argmax(x3)+1
    
    if(predicted != expected):
        numWrong += 1
        
accuracy = (samples-numWrong)/samples
print(accuracy)

# np.savetxt('w1.csv', w1, delimiter=',')
# np.savetxt('w2.csv', w2, delimiter=',')
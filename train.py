'''
Deep Learning Programming
--------------------------------------
Name: Vishal Tomar
Roll No.: 14CS30038

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import random


def sigmoid(A):
    return 1.0 / (1.0 + np.exp(-A))

def sigmoid_gradient(A):  
    return np.multiply(sigmoid(A), (1 - sigmoid(A)))

def Cost(X,Y,Theta0,Theta1):
    input_layer_size = Theta0[0,:].size - 1
    hidden_layer_size = Theta0[:,0].size
    num_labels = Theta1[:,0].size

    error = 0.0
    a1 = np.zeros((input_layer_size+1,1))
    a2 = np.zeros((hidden_layer_size+1,1))
    a3 = np.zeros((num_labels,1))
    
    m = X[:,0].size
    for i in range(m):
        a1[0] = 1.0
        a1[1:] = X[i,:].T
        a2[0] = 1
        a2[1:] = sigmoid(np.matmul(Theta0,a1))
        a3[:] = sigmoid(np.matmul(Theta1,a2)) 
        error = error + np.sum(np.multiply(-Y[i,:],np.log(a3.T)) + np.multiply(-(1-Y[i,:]),np.log(1-a3.T)))
    return error/m

def Gradient(Theta0, Theta1, trainX, trainY, lambda_=0):

    input_layer_size = Theta0[0,:].size - 1
    hidden_layer_size = Theta0[:,0].size
    num_labels = Theta1[:,0].size
    
    Delta0 = np.zeros((Theta0[:,0].size,Theta0[0,:].size))
    Delta1 = np.zeros((Theta1[:,0].size,Theta1[0,:].size))
    
    
    a1 = np.zeros((input_layer_size+1,1))
    a2 = np.zeros((hidden_layer_size+1,1))
    z2 = np.zeros((hidden_layer_size+1,1))
    a3 = np.zeros((num_labels,1))
    del2 = np.zeros((hidden_layer_size,1))
    del3 = np.zeros((num_labels,1))

    m = trainX[:,0].size
    error = 0.0;
    for i in xrange(m):
        
        a1[0] = 1;
        a1[1:] = np.matrix(trainX[i,:]).T
        z2[0] = 1
        z2[1:] = np.matmul(Theta0 , a1)
        a2[0] = 1
        a2[1:] = sigmoid(np.matmul(Theta0 , a1))
        a3[:] = sigmoid(np.matmul(Theta1 , a2 ))
        
        del3[:] = a3 - np.matrix(trainY)[i,:].T
        
        temp = np.multiply(np.matmul(Theta1.T,del3),sigmoid_gradient(z2))
        del2 = temp[1:]


        Delta0 = Delta0 + np.matmul(del2 ,a1.T)
        Delta1 = Delta1 + np.matmul(del3 ,a2.T)
        
    Grad0 = Delta0/m
    Grad1 = Delta1/m
    
    return Grad0, Grad1

def GradientDescent(Theta0, Theta1, trainX, trainY, num_iters, learning_rate, lambda_=0):
    input_layer_size = Theta0[0,:].size - 1
    hidden_layer_size = Theta0[:,0].size
    num_labels = Theta1[:,0].size
    mini_batch_size = 10000
    for i in xrange(num_iters):
        print "Iteration ",i," ) Cost : ",Cost(trainX,trainY,Theta0,Theta1)
        test(trainX,trainY,Theta0,Theta1)
        for k in xrange(0,trainX[:,0].size,mini_batch_size):
            Grad0,Grad1 = Gradient(Theta0,Theta1,trainX[k:k+mini_batch_size,:], trainY[k:k+mini_batch_size,:], lambda_)
            Theta0 = Theta0 - np.multiply(learning_rate,Grad0)
            Theta1 = Theta1 - np.multiply(learning_rate,Grad1)
    return Theta0[:,:], Theta1[:,:]


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    trainX = np.reshape(trainX,(trainX.size/784,-1))
    trainX = trainX/255.0 - 0.5
    #trainX = trainX/1000.0
    trainX = np.matrix(trainX)
    trainY = np.matrix(trainY).T
    
    m = trainX[:,0].size
    n = trainX[0,:].size
    input_layer_size = n;
    hidden_layer_size = 50;
    num_labels = 10
    
    trainY_ = np.zeros((m,num_labels))
    for i in xrange(m):
	    trainY_[i,trainY[i]] = 1
    trainY_ = np.matrix(trainY_)
	
    epsilon = 0.125
    Theta0 = np.random.rand(hidden_layer_size,input_layer_size+1)*2*epsilon - epsilon
    Theta1 = np.random.rand(num_labels,hidden_layer_size+1)*2*epsilon - epsilon
    #print "Cost : ",Cost(trainX,trainY_,Theta0,Theta1)
    num_iters = 300
    learning_rate = 1.0
    #test(trainX,trainY_,Theta0,Theta1)    
    Theta0, Theta1 = GradientDescent(Theta0, Theta1, trainX, trainY_, num_iters, learning_rate, 0)
    test(trainX,trainY_,Theta0,Theta1)
    np.savez('weights.npz', a=Theta0, b=Theta1)
    
def test(trainX,trainY,Theta0,Theta1):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    #data = np.load('weights.npz')
    #Theta0 = data['a']
    #Theta1 = data['b']
    #testX = np.reshape(testX,(testX.size/784,-1))
    #testX = testX/255.0 - 0.5
    #testX = np.matrix(testX)

    input_layer_size = Theta0[0,:].size - 1
    hidden_layer_size = Theta0[:,0].size
    num_labels = Theta1[:,0].size
    
    a1 = np.zeros((input_layer_size+1,1))
    a2 = np.zeros((hidden_layer_size+1,1))
    a3 = np.zeros((num_labels,1))
    labels = np.zeros((trainX[:,0].size,1))
    m = trainX[:,0].size
    error = 0.0;
    for i in range(m):
        a1[0] = 1;
        a1[1:] = np.matrix(trainX[i,:]).T
        a2[0] = 1
        a2[1:] = sigmoid(np.matmul(Theta0 , a1))
        a3 = sigmoid(np.matmul(Theta1 , a2 ))
        labels[i] = np.argmax(a3)
    
    accuracy = np.mean((labels == np.argmax(trainY,axis=1) )) * 100.0
    print "\nTrain accuracy: %lf%%" % accuracy
    #return labels    
    
def test2(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    data = np.load('weights.npz')
    Theta0 = data['a']
    Theta1 = data['b']
    testX = np.reshape(testX,(testX.size/784,-1))
    testX = testX/255.0 - 0.5
    testX = np.matrix(testX)

    input_layer_size = Theta0[0,:].size - 1
    hidden_layer_size = Theta0[:,0].size
    num_labels = Theta1[:,0].size
    
    a1 = np.zeros((input_layer_size+1,1))
    a2 = np.zeros((hidden_layer_size+1,1))
    a3 = np.zeros((num_labels,1))
    labels = np.zeros((testX[:,0].size,1))
    m = testX[:,0].size
    error = 0.0;
    for i in range(m):
        a1[0] = 1;
        a1[1:] = np.matrix(testX[i,:]).T
        a2[0] = 1
        a2[1:] = sigmoid(np.matmul(Theta0 , a1))
        a3 = sigmoid(np.matmul(Theta1 , a2 ))
        labels[i] = np.argmax(a3)
    
    #accuracy = np.mean((labels == np.argmax(trainY,axis=1) )) * 100.0
    #print "\nTrain accuracy: %lf%%" % accuracy
    return labels    
    
    
    
if __name__ == '__main__':
    print "Hello"
    #print sigmoid(np.array([-10,0,10]))



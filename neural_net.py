#Initial Analysis
import numpy as np
import pandas as pd

#Network Implementation
class NeuralNet(object):

    def __init__(self, in_size, hid_size, out_size):
        '''
        in_size: Dimension of input data
        hid_size: Number of hidden layer units
        out_size: number of classes(in this project is 2)
        '''
        std = 1e-4
        # Initialize to small random values
        self.W1 = std * np.random.randn(in_size, hid_size) #[D, H]
        self.W2 = std * np.random.randn(hid_size, out_size) #[H, C]
        # Initialize to zero
        self.b1 = np.zeros(hid_size) #[H,]
        self.b2 = np.zeros(out_size) #[C,]

    def loss(self, X, Y, reg): # Compute loss and gradients
        '''
        X: Input data(tranining examples)
        Y: Tranining examples labels
        reg: regularization strength
        '''
        W1 = self.W1
        b1 = self.b1
        W2 = self.W2
        b2 = self.b2
        N, D = X.shape # N: Number of traning samples, D: Dimention of input data

        # Forward propagation: compute loss
        # First layer:
        z1 = np.dot(X, W1) + b1 # [N, H]
        # ReLU:
        p1 = np.maximum(0, z1) # [N, H]
        # Second layer:
        z2 = np.dot(p1, W2) + b2 # [N, C]

        # Compute the loss: 
        # Data loss(Softmax Classifier = Softmax + Log likelihood loss):
        # Softmax
        unnormd_prob = np.exp(z2) #unnormalized probabilities matrix, shape = [N, C]
        C = len(unnormd_prob[0])
        p2 = np.copy(unnormd_prob)
        for i in range(N):
            p2[i, range(C)] = unnormd_prob[i, range(C)]/np.sum(unnormd_prob[i]) # [N, C]
        # Log likelihood loss
        data_loss = -(np.sum(np.log(p2[range(N), Y[range(N)]])))/N
        #L2 regularization loss(to W1 and W2))
        reg_loss = reg * np.sum(W1 ** 2) + reg * np.sum(W2 ** 2)
        #Loss = Data loss + Regularization loss
        loss = data_loss + reg_loss

        # Backward propagation: compute gradients
        #Backprop to Softmax
        dz2 = np.copy(p2) #shape: (N, C)
        dz2[range(N),Y[range(N)]] -=1 
        dz2 /= N
    
        dW2 = np.dot(p1.T, dz2) # [H, C]
        db2 = np.sum(dz2, axis = 0) # [C,]
        dp1 = np.dot(dz2, W2.T) # [N, H]
        dz1 = np.copy(dp1) # [N, H]
        #Backprop to ReLU
        dz1[z1 <= 0] = 0 # [N, H]
        dW1 = np.dot(X.T, dz1) # [D, H]
        db1 = np.sum(dz1, axis = 0) # [H,]
        #Gradients for L2 regularization term
        dW1_reg = 2 * reg * W1
        dW2_reg = 2 * reg * W2
    
        grads = {}
        grads['W1'] = dW1 + dW1_reg
        grads['W2'] = dW2 + dW2_reg
        grads['b1'] = db1
        grads['b2'] = db2

        return loss, grads

    def train(self, X, Y, X_test, Y_test, num_iters, learning_rate, learning_rate_decay,reg,
              batch_size, print_losses = False): #Train dataset
        '''
        X: Input data(tranining examples)
        Y: Tranining examples labels
        X_test: Testing examples
        Y_test: Testing examples labels
        num_iters: number of iterations
        learning_rate: learning rate
        learning_rate_decay: learning rate decay(used to decay LR after each epoch)
        batch_size: number of examples in a single batch
        print_losses: boolean used to print losses during training(default is False)
        '''
        # Use Mini-batch GD to optimize the parameters
        num_train = X.shape[0]
        losses = []
        train_accuracies = []
        test_accuracies = []
        
        # Create random minibatches of training data and labels
        for ite in range(num_iters):
            idx = np.random.choice(num_train, batch_size)
            X_batch = X[idx]
            Y_batch = Y[idx]
            
            # Compute loss and gradients using minibatch created above
            loss, grads = self.loss(X_batch, Y = Y_batch, reg = reg)
            
            # Update parameters
            self.W1 -= learning_rate * grads['W1']
            self.W2 -= learning_rate * grads['W2']
            self.b1 -= learning_rate * grads['b1']
            self.b2 -= learning_rate * grads['b2']

            #Check train and test accuracy and decay learning rate every epoch
            iters_per_epoch = int(num_train / batch_size)
            if ite % iters_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == Y_batch).mean()
                test_acc = (self.predict(X_test) == Y_test).mean()
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                # Decay learning rate
                learning_rate *= learning_rate_decay
                
            losses.append(loss)
            # print loss every 100 iteration
            if print_losses:
                if ite % 100 == 0:
                    print('Iteration %d: loss = %f' % (ite, loss))

        return {
            'losses': losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
        }

    def predict(self, X): #Predict labels using trained weights
        #1st layer
        z1 = np.dot(X, self.W1) + self.b1
        #ReLU
        p1 = np.maximum(0, z1)
        #2nd layer
        z2 = np.dot(p1, self.W2) + self.b2
        #predicted labels
        Y_pred = np.argmax(z2, axis=1)

        return Y_pred
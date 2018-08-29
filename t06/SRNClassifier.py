import numpy as np
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError

__author__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"

class SRNClassifier:

    # Constructor
    #
    # Inputs: alpha - learning parameter value for later training
    #         hidden_layer_size - number of neurons in the hidden layer
    #         activation - activation function to use on the hidden layer ('identity','logisti','tanh', or 'relu')
    #         max_iter - maximimum number of iterations to train for
    def __init__(self, alpha, hidden_layer_size, activation, max_iter = 500, verbose=False):
        self.alpha = alpha
        self.hidden_layer_size = int(hidden_layer_size)
        self.activation = activation
        self.max_iter = max_iter
        # Initialise the hidden state to zeros
        self.context = np.zeros(self.hidden_layer_size)
        # Don't know the size of the input and output yet
        self.num_inputs = None
        self.num_outputs = None
        self.verbose = verbose

    # Resets the model's context
    #
    # Inputs: zeroState - boolean controlling whether context is reset to zero
    #                     or to a random vector
    def reset(self, zeroState=True):
        if zeroState:
            self.context = np.zeros(self.hidden_layer_size)
        else:
            self.context = np.random.uniform(-1,1,self.hidden_layer_size)

    # Update the model using vanilla steepest gradient optimiser
    #
    # Inputs: g - gradient vector (must be same dimension as self.params)
    def _update_sg(self, g):
        self.params -= self.alpha*g

    # Update the model using adam optimiser
    #
    # Inputs: g - gradient vector (must be same dimension as self.params)
    def _update_adam(self, g):

        self._adam_t += 1
        self._adam_m *= self._adam_B1
        self._adam_m += (1.0-self._adam_B1)*g

        self._adam_v *= self._adam_B2
        self._adam_v += (1.0-self._adam_B2)*g**2

        alpha = self.alpha * np.sqrt(1-self._adam_B2**self._adam_t)/(1.0-self._adam_B1**self._adam_t)

        self.params -= alpha*self._adam_m/(np.sqrt(self._adam_v)+self._adam_eps)

    # Slice the params vector into weight matrices and bias vectors
    #
    # Inputs: params - the parameter vector
    def _slice_params(self, params):
        i = 0
        # Weight matrix connecting inputs to the hidden layer
        Wih = np.reshape(params[i:i + self.hidden_layer_size * self.num_inputs],
                              (self.hidden_layer_size, self.num_inputs))
        i += self.hidden_layer_size * self.num_inputs
        # Bias vector for the hidden layer
        bh = params[i:i + self.hidden_layer_size]
        i += self.hidden_layer_size
        # Weight matrix connecting hidden layer (from the previous input, also referred to as the context) to the
        # hidden layer of the current input
        Whh = np.reshape(params[i:i + self.hidden_layer_size ** 2],
                              (self.hidden_layer_size, self.hidden_layer_size))
        i += self.hidden_layer_size ** 2
        # Weight matrix connecting the hidden layer to the outputs
        Who = np.reshape(params[i:i + self.num_outputs * self.hidden_layer_size],
                              (self.num_outputs, self.hidden_layer_size))
        i += self.num_outputs * self.hidden_layer_size
        # Bias vector on the outputs
        bo = params[i:i + self.num_outputs]
        # Return the parpameters split into layred weight matrices and bias vectors
        return (Wih, bh, Whh, Who, bo)

    # Compute the output of the model
    #
    # Input: x - an NxM input matrix of N samples of M-dimensions each
    #
    # Returns: (y_out, y_hidden) - a tuple consisting of an NxK and NxU matrices, where K is the number of model's
    #                              outputs and U is the number of hidden units
    def _compute_forward(self, x):
        # Number of samples
        N = x.shape[0]

        # Slice the model parameters into layer weight matrices and bias vectors
        Wih, bh, Whh, Who, bo = self._slice_params(self.params)

        # Allocate memory for the hidden layer's activity
        #v_hidden = np.zeros((N, self.hidden_layer_size))
        # Allocate memory for the hidden layer's output
        y_hidden = np.zeros((N, self.hidden_layer_size))

        # Compute output one sample at a time (since we need context from the previous sample to compute the
        # output for the next sample)
        for n in range(N):
            # Compute the output of the hidden layer based on the input sample and the context
            y_hidden[n,:] = np.dot(x[n,:], Wih.transpose())
            y_hidden[n,:] += np.dot(self.context, Whh.transpose())
            y_hidden[n,:] += bh

            # Apply the activation function
            if self.activation == 'relu':
                y_hidden[n,y_hidden[n,:] < 0] = 0
            elif self.activation == 'logistic' or self.activation == 'sigmoid':
                y_hidden[n,:] = 1/(1+np.exp(-y_hidden[n,:]))
            elif self.activation == 'tanh':
                y_hidden[n,:] = np.tanh(y_hidden[n,:])
            elif self.activation != 'identity':
                raise ValueError(
                    "The activation '%s' is not supported. Supported activations are ('identity', 'logistic', 'tanh', 'relu')." % (
                        self.activation))

            # Update the context for the next sample
            self.context = y_hidden[n,:]

        # Compute the activity on
        y_out = np.matmul(y_hidden, Who.transpose())+bo
        y_out = y_out-np.expand_dims(np.max(y_out,axis=1),axis=1)
        y_out = np.exp(y_out)
        y_out = y_out/np.expand_dims(np.sum(y_out, axis=1), axis=1)

        return (y_out, y_hidden)

    # Trains the model
    #
    # Input: x - an NxM input matrix, where N is the number of samples and M is the dimension of each sample
    #        y - an NxK target output matrix, where N is the number of samples and K is the size of the labels
    def fit(self, x, y):

        # If the input is an M-dimensional vector, treat it as a 1xM matrix
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        # If the target output is a K-dimensional vector, treat it as a 1xK matrix
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=0)


        # If this function is being run for the first time, infer the size of the input and output and
        # set the model parameters accordingly
        if self.num_inputs is None:
            self.num_inputs = x.shape[1]
            self.num_outputs = y.shape[1]

            # Compute the number of parameters needed for the model
            nParams = self.hidden_layer_size*(self.num_inputs+1)+self.hidden_layer_size*(self.hidden_layer_size+1)+self.num_outputs*(self.hidden_layer_size+1)
            # Create the parameters vector and initialise all the components to a small random value
            self.params = np.random.randn(nParams)*0.1

        # Split the parameters into layered weight matrices and bias vectors
        Wih, bh, Whh, Who, bo = self._slice_params(self.params)

        # Softmax epsilon assures that no softmax output is exactly 0 or 1 (which would break the cost computation with
        # the logs
        _softmax_epsilon = 1e-8

        # Number of points in the training set
        N = x.shape[0]

        # Allocate memory for parameter gradient vector
        g_params = np.zeros(len(self.params))

        # Initialise parametes for the adam optimiser
        self._adam_t = 0
        self._adam_m = np.zeros(len(self.params))
        self._adam_v = np.zeros(len(self.params))
        self._adam_B1 = 0.9
        self._adam_B2 = 0.999
        self._adam_eps = 1e-8

        # Split the gradient parameters into layered weight matrices and bias vectors
        g_Wih, g_bh, g_Whh, g_Who, g_bo = self._slice_params(g_params)

        # Iterate over at most max_iter epochs
        for i in range(self.max_iter):

            # Reset the total cost
            J = 0
            # Reset the total error
            E = 0
            # Previous blame from the hidden layer
            e_context = np.zeros(self.hidden_layer_size)

            # Reset the model's context
            self.reset()

            # Clear the delta parameters to 0
            g_params *= 0.0

            # Compute the output of all the layers of the network
            (y_out, y_hidden) = self._compute_forward(x)

            # Make sure that no softmax output is exactly 0 or exactly 1
            y_out[y_out < _softmax_epsilon] = _softmax_epsilon
            y_out[y_out > 1.-_softmax_epsilon] = 1.-_softmax_epsilon

            # Compute the cross-entropy cost - should be ok for log(y_out) and log(1-yout) since y_out cannot be
            # 0 nor 1
            J += np.sum(np.sum(-y*np.log(y_out)-(1.-y)*np.log(1.-y_out),axis=1))
            # Compute the number of errors
            errors =  np.abs(np.argmax(y_out,axis=1)-np.argmax(y,axis=1)) != 0
            E += np.sum(errors.astype('int'))

            # Backpropagation trought time of the error, starting at the last sample
            for n in reversed(range(N)):
                # Get the output of the hidden layer
                yh = y_hidden[n,:]

                # Compute the error blame from the cross-entropy cost
                e = np.copy(y_out[n,:])
                e[y[n,:]==1.0] -= 1.0

                # Divide the error by the number of sample in the dataset
                e = e/N

                # Compute the gradient to the output weight marix and bias
                g_Who += np.outer(e,yh)
                g_bo += e

                # Compute the blame on the hidden layer output - it's the blame for this output when used as context
                # plus the blame from the output
                e = e_context+np.dot(e, Who)

                # Multiply the blame by the derivative of the activation function
                if self.activation == 'relu':
                    e[yh==0] = 0
                elif self.activation == 'logistic' or self.activation == 'sigmoid':
                    e *= yh*(1-yh)
                elif self.activation == 'tanh':
                    e *= (1-yh**2)
                elif self.activation != 'identity':
                    raise ValueError("The activation '%s' is not supported. Supported activations are ('identity', 'logistic', 'tanh', 'relu')." % (self.activation))

                # Get the context value (it's the previous output of the hidden layer) for sample n>0...and
                # it's a zero vector for the first training sample
                if n>0:
                    y_context = y_hidden[n-1,:]
                else:
                    y_context = np.zeros(self.hidden_layer_size)

                # Compute the gradient on the input to hidden layer weight matrix, the hidden layer bias vector
                # and the context to hidden (or hidden to hidden) weight matrix
                g_Wih += np.outer(e,x[n,:])
                g_bh += e
                g_Whh += np.outer(e,y_context)

                # Compute the blame on the context for the preceeding sample
                e_context = np.dot(e, Whh)

            # Update the model with the adam optimiser
            self._update_adam(g_params)

            # Compute the accuracy of the model
            E = 1-float(E)/float(N)
            # If doing perfectly, stop training
            if E==1.0:
                print("Training reached perfect score. Stopping")
                break

            # If the first, last, or one of the update epoch, show the progress
            if self.verbose:
                print("Iteration %d, loss = %.6e, score=%.6f " % (i+1,J,E))

    # Check if the model has been fitted
    def check_is_fitted(self, msg=None):
        if self.num_inputs is None:
            if msg is None:
                msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                       "appropriate arguments before using this method.")
            raise NotFittedError(msg % {'name': type(self).__name__})

    # Compute the output of the model
    #
    # Input: x - an NxM matrix of N input of M-dimensons each, or a M-dimensional vector representing a signle input
    #        categorical - if set to True, will otuput a zero-one vector, if set to fals, will otuput the softmax
    #                      probability distribution over the outputs
    #
    # Returns: y - an NxK matrix of outputs, or a K-dimensional vector represeting a single output
    def predict(self,x,categorical=True):

        self.check_is_fitted()

        # Check if input is just a vector...
        _vec_input = False
        if len(x.shape)==1:
            #...if so, convert it to a 1xM matrix
            x = np.expand_dims(x,axis=0)
            _vec_input = True

        # Compute the softmax output of the model
        y = self._compute_forward(x)[0]

        # If categorical, for each sample pick the output with highest probability and mark it as 1, leaving rest
        # set to zero
        if categorical:
            y = np.argmax(y,axis=1)

            y = np.expand_dims(y, axis=1)
            enc = preprocessing.OneHotEncoder(n_values=self.num_outputs)
            enc.fit(y)
            y = enc.transform(y).toarray()

        # If input was in vector format, convet the matrix otuput to a vector
        if _vec_input:
            y = np.squeeze(y,axis=0)

        return y














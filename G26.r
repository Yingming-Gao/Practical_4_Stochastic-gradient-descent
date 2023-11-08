# The names and university user names of each member of the 26th group:
## Jin Lu(s2595995); Mingxi Liu(s2400115); Yingming Gao(s2601869)

# The address of our github repo:
## https://github.com/Yingming-Gao/Practical_4_Stochastic-gradient-descent.git

# The contributions of our team members:
## 

# Overview:
##

# Details:
##

# Define the function 'netup' to initialize the network.
netup <- function(d){
  ## Argument d is a vector of number of nodes in each layer of a network.
  ## Set a seed for random process.
  set.seed(0)
  
  ## Initialize an empty network list for storing hole network.
  nn <- list()
  
  ## Initialize empty lists for nodes values, weights and offset parameters.
  ## h is a list of nodes for each layer.
  nn$h <- list()
  ## W is a list of weight matrices. 
  nn$W <- list()
  ## b is a list of offset vectors.
  nn$b <- list()
  
  ## Initialize node values of each layer with zero values.
  ## h[[l]] should be a vector contain the node values for layer l.
  for (l in 1:length(d)) {nn$h[[l]] <- matrix(numeric(d[l]), ncol = 1)}
  
  ## Initialize weights and offsets with random values between 0 and 0.2.
  for (l in 1:(length(d) - 1)) {
    ## W[[l]] is the weight matrix linking layer l to layer l+1.
    ## Its dimension should be d[l+1] * d[l].
    nn$W[[l]] <- matrix(runif(d[l]*d[l+1], 0, 0.2), nrow = d[l+1], ncol = d[l])
    
    ## b[[l]] is the offset vector linking layer l to layer l+1.
    ## Its dimension should be d[l+1] * 1.
    nn$b[[l]] <- matrix(runif(d[l+1], 0, 0.2), ncol = 1)
  }
  
  ## This function returns a list representing the network.
  return(nn)
}


# Define the function 'forward'.
forward <- function(nn,inp){
  ## nn is a network list as returned by netup.
  ## inp is a vector of input values for the first layer.
  
  # Initialize the first layer's values with input inp
  nn$h[[1]] <- inp
  # Forward pass: compute the values for each layer
  for (i in 1:(length(nn$W))) {
    # Calculate the values of linearity
    h_i <- nn$h[[i]] %*% nn$W[[i]] + nn$b[[i]] 
    # Apply ReLU activation function
    h_values <- pmax(0, h_i) 
    # Store the values in h for the next layer
    nn$h[[i + 1]] <- h_values  
  }
  return(nn)
  ## This function compute the remaining node values implied by inp.
  ## Then return the updated network list
}


# Define the function 'backward'.
backward <- function(nn,k){
  ## nn is the undated network from 'forward'.
  ## k is the class.
  ## This function computes the derivatives of the corresponding loss.
  
  
  
  return(list(dh=dh,dW=dW,db=db))
  ## Return the list of derivatives w.r.t. the nodes, weights and offsets.
}

# Define the function 'train'.
train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  ## This function train the network, nn, given input data in the matrix inp 
  ## and corresponding labels (1, 2, 3 . . . ) in vector k. 
  ## eta is the step size η.
  ## mb is the number of data to randomly sample to compute the gradient. 
  ## nstep is the number of optimization steps to take.
  
  return(nn)
  # Return the trained network.
}

# Finally, train a 4-8-7-3 network to classify irises to species 
# based on the 4 characteristics given in the iris dataset in R. 


# Classify the test data to species according to the class predicted.
# Compute the misclassification rate.

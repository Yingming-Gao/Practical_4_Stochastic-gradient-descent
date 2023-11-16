# The names and university user names of each member of the 26th group:
## Jin Lu(s2595995); Yingming Gao(s2601869); Mingxi Liu(s2400115); 

# The address of our github repo:
## https://github.com/Yingming-Gao/Practical_4_Stochastic-gradient-descent.git

# The contributions of our team members:
## Jin Lu(30%)--Takes charge of initialization coding in this model.
## Yingming Gao(30%)--Plays a crucial role in debugging and determining the final version.
## Mingxi Liu(40%)--Responsible for main function coding. Contributes to debugging efforts.

# Overview:

## This model implements a simple neural network for classification tasks, 
## trained using stochastic gradient descent. 
## The network adjusts its weight and bias parameters through a series of functions: 
## forward propagation, backward propagation, and a training function. 
## The structure and initialization of the network are defined by the netup function. 
## Finally, the code is applied to the Iris dataset for training and testing the network, 
## and the misclassification rate is calculated.

## Here are the detailed steps:

# Details:
## 1. Set up the network structure (netup function). This function takes a vector as input, 
##    with each element representing the number of nodes in each layer of the network.
## 2. Execute forward propagation (forward function). This function takes a network 
##    (returned by the netup function) and input values for the input layer, 
##    calculates the node values for the remaining layers based on these inputs, 
##    and returns the updated network.
## 3. Execute backward propagation (backward function). This function calculates 
##    the derivative of the loss function with respect to output class k and adds 
##    the derivative values to the network list.
## 4. Train the network (train function). This function trains the network by using iris data, 
##    and adjusts the network parameters to minimize the loss using stochastic gradient descent.
## 5. Test the network and calculate the misclassification rate.

# Define the function 'netup' to initialize the network.
netup <- function(d){
  ## Argument d is a vector of number of nodes in each layer of a network.
  ## Set a seed for random process.
  set.seed(0)
  
  ## Initialize an empty network list for storing hole network.
  nn <- list()
  
  ## Initialize node values of each layer with zero values.
  ## h[[l]] should be a vector contain the node values for layer l.
  for (l in 1:length(d)) {nn$h[[l]] <- numeric(d[l])}
  
  ## Initialize weights and offsets with random values between 0 and 0.2.
  for (l in 1:(length(d) - 1)) {
    ## W[[l]] is the weight matrix linking layer l to layer l+1.
    ## Its dimension should be d[l+1] * d[l].
    nn$W[[l]] <- matrix(runif(d[l]*d[l+1], 0, 0.2), nrow = d[l+1], ncol = d[l])
    
    ## b[[l]] is the offset vector linking layer l to layer l+1.
    ## Its dimension should be d[l+1] * 1.
    nn$b[[l]] <- runif(d[l+1], 0, 0.2)
  }
  
  ## This function returns a list representing the network.
  return(nn)
}


# Define the function 'forward' to calculate node values.
forward <- function(nn,inp){
  ## nn is a network list as returned by 'netup' function.
  ## inp is a vector of input values for the first layer.
  
  # Assign the input values to the first layer.
  nn$h[[1]] <- inp
  
  # Forward pass: compute the values in subsequent layers.
  for (l in 1:(length(nn$h)-1)){
    # Calculate the l+1 layer by using W_l * h_l + b_l.
    h_l_1 <- nn$W[[l]] %*% nn$h[[l]] + nn$b[[l]] 
    
    # Apply ReLU activation function: max(0,h₁₊₁)
    h_l_1 <- pmax(0, h_l_1) 
    
    # Store the calculated values as a vector in h₁₊₁ layer.
    nn$h[[l+1]] <- h_l_1
  }
  
  ## This function returns the updated network list.
  return(nn)
}


# Define the function 'backward' to compute the derivatives of loss.
backward <- function(nn,k){
  ## nn is the undated network from 'forward'.
  ## k is the class.
  
  ## Get the number of layers, including input layer
  L <- length(nn$h) 

  # Initialize derivatives of loss w.r.t the output layer
  dh_l_1 <- exp(nn$h[[L]]) / sum(exp(nn$h[[L]]))
  # The derivative w.r.t the node of class k is different
  dh_l_1[k] <- dh_l_1[k] - 1

  
  # Back-propagation process
  for (l in (L-1):1) {
    # Due to ReLU transformation, node value could be 0
    # The relevant derivatives should be set to 0
    dh_l_1 <- dh_l_1 * (nn$h[[l+1]] > 0)
    dh_l <- t(nn$W[[l]]) %*% dh_l_1
    # Compute the derivative of h  
    nn$dh[[l]] <- dh_l
    # Compute the derivative of W
    nn$dW[[l]] <- dh_l_1 %*% t(nn$h[[l]])
    # Compute the derivative of b
    nn$db[[l]] <- dh_l_1
    # Update the d
    dh_l_1 <- dh_l
  }
  
  return(nn)
  ## Update the list with derivatives w.r.t. the nodes, weights and offsets.
}


# Define the function 'train'.
train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  ## This function trains the neural network, nn, with the given input data in the matrix inp,
  ## and corresponding labels (1, 2, 3 . . . ) in the vector k.
  ## eta is the step size η.
  ## mb is the number of data to randomly sample for computing the gradient.
  ## nstep is the number of optimization steps to take.
  
  ## Set a seed to ensure reproducibility in random processes.
  set.seed(0)
  
  # Get the number of rows in the input data
  n <- nrow(inp)
  
  for (step in 1:nstep) {
    # Sample a minibatch from the input data
    idx <- sample(1:n, mb)
    inp_mb <- inp[idx, ]
    k_mb <- k[idx]
    
    # Initialize a list to store the gradients for the entire minibatch
    batch <- list()
    for (l in 1:(length(nn$h) - 1)) {
      batch$dW[[l]] <- matrix(0, nrow = nrow(nn$W[[l]]), ncol = ncol(nn$W[[l]]))
      batch$db[[l]] <- rep(0, length(nn$b[[l]]))
    }
    
    # Compute the gradients for each data point in the minibatch
    for (i in 1:mb) {
      nn <- forward(nn, as.vector(inp_mb[i, ])) # Propagate the input forward through the network
      nn <- backward(nn, k_mb[i]) # Compute the gradients by backpropagation
      
      # Accumulate the gradients for the entire minibatch
      for (l in 1:(length(nn$h) - 1)) {
        batch$dW[[l]] <- batch$dW[[l]] + nn$dW[[l]]
        batch$db[[l]] <- batch$db[[l]] + nn$db[[l]]
      }
    }
    
    # Update the network parameters using the mean of the batch gradients
    for (l in 1:(length(nn$h) - 1)) {
      nn$W[[l]] <- nn$W[[l]] - eta * batch$dW[[l]] / mb # Update the W
      nn$b[[l]] <- nn$b[[l]] - eta * batch$db[[l]] / mb # Update the b
    }
    
  }
  
  return(nn) # Return the trained network.
}

# Assume compute_loss function is defined to compute the loss of the model
# based on the current parameters and the entire dataset.

# Finally, train a 4-8-7-3 network to classify irises to species 
# based on the 4 characteristics given in the iris dataset in R. 
# import the iris data
data ("iris")

# Divide the iris data into training data and test data
test_input <- as.matrix(iris[seq(5, nrow(iris), by = 5), -ncol(iris)])
training_input <- as.matrix(iris[-seq(5, nrow(iris), by = 5), -ncol(iris)])

# Ensure test values are numeric
test_out <- as.integer(iris[seq(5, nrow(iris), by = 5), ncol(iris)])
training_out <- as.integer(iris[-seq(5, nrow(iris), by = 5), ncol(iris)])
  
# Initialize the network
d = matrix(c(4,8,7,3),ncol = 1)
nn = netup(d)
# Train the network
nn = train(nn,training_input,training_out)

# Classify the test data to species according to the class predicted.
# Compute the misclassification rate.
pre = matrix(0,nrow = 30, ncol = 1)
for (i in 1:30){
  a = as.vector(test_input[i,])
  
  prediction = forward(nn,a)$h[[4]]
  pre[i,1] = which.max(prediction)
}

# Print the misclassification rate
misclassification_rate <- sum(pre != test_out) / length(test_out)
cat("Misclassification Rate:", misclassification_rate, "\n")


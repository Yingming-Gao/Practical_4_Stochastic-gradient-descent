# The names and university user names of each member of the 26th group:
## Jin Lu(s2595995); Yingming Gao(s2601869); Mingxi Liu(s2400115); 

# The address of our github repo:
## https://github.com/Yingming-Gao/Practical_4_Stochastic-gradient-descent.git

# The contributions of our team members:
## Member 1(30%):Jin Lu--Takes charge of initialization coding in this model, provides valuable ideas during discussions.
## Member 2(30%):Yingming Gao--Plays a crucial role in debugging and determining the final version.
## Member 3(40%):Mingxi Liu--Responsible for main function coding. Contributes to debugging efforts.

# Overview:
## This model implements a simple neural network for classification，and train it using stochastic gradient descent.
## The network adjusts its weight and offset parameters through forward function, backward function and train function.
## The structure and initialization of the network are defined by the netup function.
## Finally, the code uses iris data set to train and test the network, and calculates the misclassification rate.

# Details:
## 1) Function netup: Initialize the network. Parameter d is a vector, which represents the number of nodes in each layer of the network.
##    This function creates an empty network with node values (h), weights (w) and offsets (b). 
##    The weights and offsets are initialized with random numbers between 0 and 0.2.
## 2) Function forward: Forward propagation. This function accepts the input value inp of the network nn and updates the node value 
##    of the network through layer-by-layer calculation. ReLU transform is applied to the output of each layer.
## 3) Function backward: This function calculates the derivatives of the loss respect to the output class' k'. This includes calculating the
##    derivatives of nodes, weights and offsets relative to the network, and adding these derivatives to the network structure as lists dh, dW and db.
## 4) Function train：This function receives the rows of the input data matrix inp as input, and uses the corresponding labels (such as 1, 2, 3, etc.) in the vector K for training.

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
    
    nn$dh[[l]] <- dh_l
    nn$dW[[l]] <- dh_l_1 %*% t(nn$h[[l]])
    nn$db[[l]] <- dh_l_1
    
    dh_l_1 <- dh_l
  }
  
  return(nn)
  ## Update the list with derivatives w.r.t. the nodes, weights and offsets.
}


# Define the function 'train'.
train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  ## This function train the network, nn, given input data in the matrix inp 
  ## and corresponding labels (1, 2, 3 . . . ) in vector k. 
  ## eta is the step size η.
  ## mb is the number of data to randomly sample to compute the gradient. 
  ## nstep is the number of optimization steps to take.
  
  ## Set a seed for random process.
  set.seed(0)
  
  n <- nrow(inp)
  
  for (step in 1:nstep) {
    # Sample minibatch
    idx <- sample(1:n, mb)
    inp_mb <- inp[idx, ]
    k_mb <- k[idx]
    
    # Initialize a list to store batch gradients
    batch <- list()
    for (l in 1:(length(nn$h) - 1)) {
      batch$dW[[l]] <- matrix(0, nrow = nrow(nn$W[[l]]), ncol = ncol(nn$W[[l]]))
      batch$db[[l]] <- rep(0, length(nn$b[[l]]))
    }
    
    # Compute gradients for the entire minibatch
    for (i in 1:mb) {
      nn <- forward(nn, as.vector(inp_mb[i, ]))
      nn <- backward(nn, k_mb[i])
      
      for (l in 1:(length(nn$h) - 1)) {
        batch$dW[[l]] <- batch$dW[[l]] + nn$dW[[l]]
        batch$db[[l]] <- batch$db[[l]] + nn$db[[l]]
      }
    }
    
    # Update parameters
    for (l in 1:(length(nn$h) - 1)) {
      nn$W[[l]] <- nn$W[[l]] - eta * batch$dW[[l]] / mb
      nn$b[[l]] <- nn$b[[l]] - eta * batch$db[[l]] / mb
    }
    
  }
  
  return(nn)
  # Return the trained network.
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

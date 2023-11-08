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

# Define the function 'netup'.
netup <- function(d){
  ## argument d is a vector of number of nodes in each layer of a network
  
  
  
  
  
  return(list(h=h,W=W,b=b))
  ## This function returns a list representing the network
  ## h a list of nodes for each layer. 
  ## W a list of weight matrices. 
  ## b a list of offset vectors.
}

# Define the function 'forward'.
forward <- function(nn,inp){
  ## nn is a network list as returned by netup.
  ## inp is a vector of input values for the first layer.
  
  
  
  
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

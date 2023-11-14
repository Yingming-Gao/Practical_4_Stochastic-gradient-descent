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
    nn$h[[l+1]] <- matrix(h_l_1, ncol = 1)
  }
  
  ## This function returns the updated network list.
  return(nn)
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

backward <- function(nn, k) {
  L <- length(nn$W) + 1  # number of layers, including input
  n <- length(nn$h[[L]]) # number of output nodes
  
  # Step 2: Initialize derivative of loss wrt output layer
  dL_dh <- exp(nn$h[[L]]) / sum(exp(nn$h[[L]]))
  dL_dh[k] <- dL_dh[k] - 1
  
  # Backpropagation
  for (l in (L - 1):2) {
    dh_next <- dL_dh
    dh_current <- nn$W[[l - 1]] %*% dh_next
    
    relu_grad <- as.numeric(nn$h[[l]] > 0)
    dL_dh <- relu_grad * dh_current
    
    nn$dh[[l - 1]] <- dL_dh
    nn$dW[[l - 1]] <- dh_next %*% t(nn$h[[l - 1]])
    nn$db[[l - 1]] <- dh_next
  }
  
  return(nn)
}

train <- function(nn, inp, k, eta = 0.01, mb = 10, nstep = 10000) {
  n <- nrow(inp)
  
  for (step in 1:nstep) {
    # Sample minibatch
    idx <- sample(1:n, mb)
    inp_mb <- inp[idx, ]
    k_mb <- k[idx]
    
    # Initialize gradients
    for (l in 1:(length(nn$h) - 1)) {
      nn$dh[[l]] <- matrix(0, nrow = ncol(nn$W[[l]]), ncol = mb)
      nn$dW[[l]] <- matrix(0, nrow = nrow(nn$W[[l]]), ncol = ncol(nn$W[[l]]))
      nn$db[[l]] <- rep(0, ncol(nn$W[[l]]))
    }
    
    # Compute gradients for each sample in minibatch
    for (i in 1:mb) {
      nn <- forward(nn, inp_mb[i, ])
      nn <- backward(nn, k_mb[i])
    }
    
    # Update parameters
    for (l in 1:(length(nn$h) - 1)) {
      nn$W[[l]] <- nn$W[[l]] - eta * nn$dW[[l]] / mb
      nn$b[[l]] <- nn$b[[l]] - eta * nn$db[[l]] / mb
    }
  }
  
  return(nn)
}




##修改版backward
backward <- function(nn, k) {
  # 初始化空列表以存储梯度
  nn$dh <- vector("list", length(nn$h))
  nn$dW <- vector("list", length(nn$W))
  nn$db <- vector("list", length(nn$b))
  d <- vector("list", length(nn$h))
  # 计算损失函数对于 hL_j 的导数
  q <- exp(nn$h[[length(nn$h)]])
  nn$dh[[length(nn$dh)]] <- q / sum(q)
  nn$dh[[length(nn$dh)]][k] <- nn$dh[[length(nn$dh)]][k] - 1
  
  # 对最后一层的导数进行ReLU判断
  d[[length(nn$dh)]] <- pmax(0, nn$dh[[length(nn$dh)]]) # 
  
  # 通过反向传播计算 Li 对于所有其他 hl_j 的导数
  for (l in (length(nn$h) - 1):1) {
    nn$dh[[l]] <-  (t(nn$W[[l]]) %*% d[[l + 1]]) 
    nn$dW[[l]] <- d[[l + 1]] %*% t(nn$h[[l]])
    nn$db[[l]] <- d[[l + 1]]
    d[[l]] <- pmax(0,t(nn$W[[l]]) %*% d[[l+1]] )
  }
}



train <- function(nn, inp, k, eta=.01, mb=10, nstep=10000) {
  for (step in 1:nstep) {
    # 随机选择 mb 个数据点
    indices <- sample(1:nrow(inp), mb)
    sampled_inp <- inp[indices,]
    sampled_k <- k[indices]
    
    # 执行前向和后向传播
    for (i in 1:mb) {
      nn <- forward(nn, sampled_inp[i,])
      nn <- backward(nn, sampled_k[i])
      # 计算梯度的平均值并更新参数
      for (l in 1:(length(nn$W) - 1)) {
        nn$W[[l]] <- nn$W[[l]] - eta * nn$dW[[l]]
        nn$b[[l]] <- nn$b[[l]] - eta * nn$db[[l]]
      }
    }
  }
    
   
  return(nn)
  
  
  
}






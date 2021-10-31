# The Baum-Welch algorihtm

myBW = function(x, para, n.iter = 100){
  # Input:
  # x: T-by-1 observation sequence
  # para: initial parameter value
  # Output updated para value (A and B; we do not update w)

  for(i in 1:n.iter){
    para = BW.onestep(x, para)
  }
  return(para)
}


####### Your function BW.onestep, in which we operate the E-step and M-step for one iteration, should look as follows.

BW.onestep = function(x, para){
  # Input: 
  # x: T-by-1 observation sequence
  # para: mx, mz, and current para values for
  #    A: initial estimate for mz-by-mz transition matrix
  #    B: initial estimate for mz-by-mx emission matrix
  #    w: initial estimate for mz-by-1 initial distribution over Z_1
  # Output the updated parameters after one iteration
  # We DO NOT update the initial distribution w
  
  T = length(x)
  mz = para$mz
  mx = para$mx
  A = para$A
  B = para$B
  w = para$w
  alp = forward.prob(x, para)
  beta = backward.prob(x, para)
  
  myGamma = array(0, dim=c(mz, mz, T-1))
  #######################################
  ## YOUR CODE: 
  ## Compute gamma_t(i,j) P(Z[t] = i, Z[t+1]=j), 
  ## for t=1:T-1, i=1:mz, j=1:mz, 
  ## which are stored in an array, myGamma
  
  
  #######################################
  
  # M-step for parameter A
  #######################################
  ## YOUR CODE: 
  ## A = ....
  #######################################
  
  # M-step for parameter B
  #######################################
  ## YOUR CODE: 
  ## B = ....
  #######################################
  
  para$A = A
  para$B = B
  return(para)
}



# forward-backward probabilities

forward.prob = function(x, para){
  # Output the forward probability matrix alp 
  # alp: T by mz, (t, i) entry = P(x_{1:t}, Z_t = i)
  T = length(x)
  mz = para$mz
  A = para$A
  B = para$B
  w = para$w
  alp = matrix(0, T, mz)
  
  # fill in the first row of alp
  alp[1, ] = w * B[, x[1]]
  # Recursively compute the remaining rows of alp
  for(t in 2:T){
    tmp = alp[t-1, ] %*% A
    alp[t, ] = tmp * B[, x[t]]
  }
  return(alp)
}

backward.prob = function(x, para){
  # Output the backward probability matrix beta
  # beta: T by mz, (t, i) entry = P(x_{1:t}, Z_t = i)
  T = length(x)
  mz = para$mz
  A = para$A
  B = para$B
  w = para$w
  beta = matrix(1, T, mz)
  
  # The last row of beta is all 1.
  # Recursively compute the previous rows of beta
  for(t in (T-1):1){
    tmp = as.matrix(beta[t+1, ] * B[, x[t+1]])  # make tmp a column vector
    beta[t, ] = t(A %*% tmp)
  }
  return(beta)
}



### Viterbi algorithm


myViterbi = function(x, para){
  # Output: most likely sequence of Z (T-by-1)
  T = length(x)
  mz = para$mz
  A = para$A
  B = para$B
  w = para$w
  log.A = log(A)
  log.w = log(w)
  log.B = log(B)
  
  # Compute delta (in log-scale)
  delta = matrix(0, T, mz) 
  # fill in the first row of delta
  delta[1, ] = log.w + log.B[, x[1]]
  
  #######################################
  ## YOUR CODE: 
  ## Recursively compute the remaining rows of delta
  #######################################
  
  # Compute the most prob sequence Z
  Z = rep(0, T)
  # start with the last entry of Z
  Z[T] = which.max(delta[T, ])
  
  #######################################
  ## YOUR CODE: 
  ## Recursively compute the remaining entries of Z
  #######################################
  
  return(Z)
}



# Test your function

data = scan("coding4_part2_data.txt")

mz = 2
mx = 3
ini.w = rep(1, mz); ini.w = ini.w / sum(ini.w)
ini.A = matrix(1, 2, 2); ini.A = ini.A / rowSums(ini.A)
ini.B = matrix(1:6, 2, 3); ini.B = ini.B / rowSums(ini.B)
ini.para = list(mz = 2, mx = 3, w = ini.w,
                A = ini.A, B = ini.B)

myout = myBW(data, ini.para, n.iter = 100)
myout.Z = myViterbi(data, myout)
myout.Z[myout.Z==1] = 'A'
myout.Z[myout.Z==2] = 'B'



# Results from HMM
library(HMM)
hmm0 =initHMM(c("A", "B"), c(1, 2, 3),
              startProbs = ini.w,
              transProbs = ini.A, 
              emissionProbs = ini.B)
Rout = baumWelch(hmm0, data, maxIterations=100, delta=1E-9, pseudoCount=0)
Rout.Z = viterbi(Rout$hmm, data)


# Compare two results
options(digits=8)
options()$digits

# Compare estimates for transition prob matrix A
myout$A
Rout$hmm$transProbs

# Compare estimates for emission prob matrix B
myout$B
Rout$hmm$emissionProbs

# Compare the most probable Z sequence.
cbind(Rout.Z, myout.Z)[c(1:10, 180:200), ]

sum(Rout.Z != myout.Z)






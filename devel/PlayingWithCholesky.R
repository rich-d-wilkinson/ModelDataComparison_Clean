k <- function(x,xp, lambda=1){
  exp(-abs(x-xp)/lambda)
}

x <- seq(0,1,length.out = 100)

K <- matrix(nr=length(x), nc=length(x))

for(i in 1:length(x)){
  for(j in 1:length(x)){
    K[i,j] <- k(x[i],x[j])
  }
}

K2 <- matrix(nr=length(x), nc=length(x))

for(i in 1:length(x)){
  for(j in 1:length(x)){
    K2[i,j] <- k(x[i],x[j], lambda=2)
  }
}

Kchol <- chol(K)
library(mvtnorm)

#set.seed(1)
xsmp <- rmvnorm(100, sigma=K)

dmvnorm(xsmp, sigma=K, log=T)

#library(pracma)

thin <- function(x, K=NA, output=NA, thinby){
  xthin = x[seq(1,length(x),thinby)]
  Kthin = K[seq(1,length(x),thinby),seq(1,length(x),thinby)]
  outputthin = output[,seq(1,length(x),thinby)]
  return(list(xthin, Kthin, outputthin))
}
thinby=2
tmp = thin(x, K, xsmp, thinby=thinby)
xthin = tmp[[1]]
Kthin = tmp[[2]]
outputtin=tmp[[3]]

likfull <- dmvnorm(xsmp, sigma=K, log=T)
likthin <- dmvnorm(outputtin, sigma=Kthin, log=T)
print(order(likfull))
print(order (likthin))
print(cor(likfull, likthin, method = 'spearman'))

if(FALSE){
round(likfull-max(likfull))
round(likthin-max(likthin))

# So thinning does not preserve likelihood structure. Is it worse for misspecified models?

# Generate samples from the smoother covariance,
xsmp2 <- rmvnorm(100, sigma=K2)
tmp = thin(x, K, xsmp2, thinby=thinby)
Kthin = tmp[[2]]
outputtin2=tmp[[3]]

# evaluate likelihood with the coarser one.
likfull <- dmvnorm(xsmp2, sigma=K, log=T)
likthin <- dmvnorm(outputtin2, sigma=Kthin, log=T)

print('misspecified covariance')
print(order(likfull))
print(order (likthin))
print(cor(likfull, likthin, method = 'spearman'))

## Conclusions
# Likelihood score is not particularly robust to thinning of the data and covariance.
# The more you thin, the worse the problem seems. Doesn't seem to get noticeably 
# worse with misspecification.

##################################
# Distribution of log score.
x <- seq(0,1,length.out = 100)
K <- matrix(nr=length(x), nc=length(x))
for(i in 1:length(x)){
  for(j in 1:length(x)){
    K[i,j] <- k(x[i],x[j],1)}}

xsmp <- rmvnorm(100, sigma=K)
matplot(t(xsmp), type='l')
dmvnorm(xsmp, sigma=K, log=T)
dmvnorm(rep(0, dim(K)[1]), sigma=K, log=T)
dmvnorm(rep(1, dim(K)[1]), sigma=K, log=T)
dmvnorm(rep(5, dim(K)[1]), sigma=K, log=T)
dmvnorm(rep(100, dim(K)[1]), sigma=K, log=T)

}
# Conclusions
# - smooth things looks nice to the log likelihood
# - subsampling is complicated as it can change what is important in the GCM runs
# - correlations over small distances are determining to some extent which runs we favour
# - this is probably a bad idea.
# what would be better?}
# Code used to estimate optrdd in R and to compare with Julia output
data <- read.csv("data/cleaned/lee.csv")
X = data$x
Y= data$y
c= 0
M=14.28
W = as.numeric(X>c)
model <-  optrdd::optrdd(X,Y,W,
                         max.second.derivative=M,verbose=F,estimation.point=c,
                         use.homoskedatic.variance = T,
                         try.elnet.for.sigma.sq=F, use.spline=F, optimizer='mosek')

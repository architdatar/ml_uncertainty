# Shows attempts to reproduce approach by Greg Snow 
# (https://stats.stackexchange.com/questions/56895/do-the-predictions-of-a-random-forest-model-have-a-prediction-interval).


# Set up the simulation.
library(randomForest)

set.seed(1)

x1 <- rep(0:1, each=500)
x2 <- rep(0:1, each=250, length=1000)

y <- 10 + 5*x1 + 10*x2 - 3*x1*x2 + rnorm(1000)

# Test data
newdat <- expand.grid(x1=0:1, x2=0:1)


fit <- randomForest(y ~ x1 + x2, ntree=1001, keep.inbag=TRUE)

pred.rf <- predict(fit, newdat, predict.all=TRUE)

# Confidence interval
# ====================
pred.rf.int <- apply(pred.rf$individual, 1, function(x) {
  c(mean(x), sd(x), mean(x) + c(-1, 1) * sd(x), 
  quantile(x, c(0.025, 0.975)))
})

t(pred.rf.int)

# Prediction interval
# ====================

# 1. Attempt to reproduce the predicted values for a sample.
# Prediction considering train data as test data.
pred.rf.train <- predict(fit2, cbind(x1, x2), predict.all=TRUE)

# Estimated value for the first sample.
pred.rf.train$aggregate[1]
# Ground truth
fit$predicted[1]

# 2. Attempt to reproduce MSE
# Expected MSE:
fit$mse
# Attempt 1: For tree k : $$ \sigma^2_k = \frac{1}{N}\sum_{i=1}^N(y_{ik} - y_i)^2 $$
mse.1 <- sapply(1:1001, 
    function(x){
        mean((pred.rf.train$individual[, x] - y)^2)
        }
        )
# Note: In the randomForest library implementation, the MSEs are computed cumulatively.
# So, we also compute them cumulatively to compare. 
mse.1.cum <- cumsum(mse.1) / seq_along(mse.1)

# Attempt 2: Only consider out-of-bag predictions for MSEs.
oob.mask <- fit$inbag == 0
oob.mse <- sapply(1:1001, function(x){
    mean((pred.rf.train$individual[, x][oob.mask[, x]] - y[oob.mask[,x]])^2)
    })
oob.mse.cum <- cumsum(oob.mse) / seq_along(oob.mse)

# 3. Computing coverage from Approach by Greg Snow.
pred.rf.int2 <- sapply(1:1000, function(i) {
    tmp <- pred.rf.train$individual[i, ] + rnorm(1001, 0, sqrt(fit$mse))
    quantile(tmp, c(0.025, 0.975)) })
df <- t(pred.rf.int2)
coverage <- sum((df[, 1] < y) & (df[, 2] > y)) / length(y)

# Coverage came out to 0.997 which is higher than the 95% confidence level expected
# in this example. 


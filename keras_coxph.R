#########################
##  keras implementation of Cox model
##  Byron Smith
##
##  This code is used to run a convolutional neural network on a dataset from
##  the 'lifelines' module in python.

rm(list=ls())
library(keras)
library(survival)
library(abind) # For array binding
library(tensorflow)
library(reticulate)

# indir <- "Your Input Directory" # Change to use
# outdir <- "Your Output Directory" # Change to use

data1 <- read.csv(paste0(indir, "kidtx.csv"))

set.seed(4374532)
batch_size=nrow(data1)
remove_ties <- FALSE
N <- nrow(data1)

##  Here I'll be using the reticulate package to load the partial likelihood loss
##  because R has a hard time going back and forth between object types.

repl_python()
import tensorflow as tf
import keras.backend as K
def neg_log_pl(y_true, y_pred):
  # Sort by survival time (descending) so that
  # - If there are no tied survival times, the risk set
  #   for event i is individuals 0 through i
  # - If there are ties, and time[i - k] through time[i]
  #   represent all times equal to time[i], then the risk set
  #   for events i - k through i is individuals 0 through i
  sorting = tf.nn.top_k(y_true[:, 0], k = 863)
  time = K.gather(y_true[:, 0], indices = sorting.indices)
  xbeta = K.gather(y_pred[:, 0], indices = sorting.indices)
  risk = K.exp(xbeta)
  # For each set of tied survival times, put the sum of the
  # corresponding risk (exp[x * beta]) values at the first
  # position in the sorted array of times while setting other
  # positions to 0 so that the cumsum operation will result
  # in each of the positions having the same sum of risks
  for i in range(time.shape[0] - 1, 0, -1):
    # Going from smallest survival times to largest
    if time[i] == time[i - 1]:
    # Push risk to the later time (earlier in array position)
      risk[i - 1] = risk[i - 1] + risk[i]
      risk[i] = 0
  event = K.gather(y_true[:, 1], indices = sorting.indices)
  denom = K.cumsum(risk)
  terms = xbeta - K.log(denom)
  loglik = K.cast(event, dtype = terms.dtype) * terms
  return -K.sum(loglik)


exit




# Create the 'x' and  'y' matrix ---------------------------------------------------

y_mat <- data1[,c("time", "death")]
x_train <- data1[,c("age", "black_male", "white_male", "black_female")]

y_mat2 <- array(as.matrix(y_mat), dim=c(863, 2))
x_train2 <- array(as.matrix(x_train), dim=c(863,4))


# Test the functions and matrix -------------------------------------------

##  Try it out real quick to see what happens!
## In order to run this code, you must adjust the batch_size
sf1 <- coxph(Surv(time, death)~., data=data1, ties = "breslow")
preds <- as.matrix(data1[,-c(1,2)]) %*% coef(sf1)

summary(sf1)$loglik[2]

k_get_value(py$neg_log_pl(y_true=y_mat2, y_pred=preds))

# Now put the model together ----------------------------------------------

##  To creat the model we use a sequential series of 'filters' that will be applied to input images.
##  The output should always be a column vector with the dimensions (?,1)

model <- keras_model_sequential()
model %>% layer_dense(units=1, use_bias=FALSE, activation="linear", input_shape=c(4))


summary(model) # This describes the model

##  Compile the model

model %>% compile(
  loss=py$neg_log_pl,
  optimizer = optimizer_adam() # You can choose a whole series of optimizers
)


# A new strategy ----------------------------------------------------------


model %>% fit(
  x_train2, y_mat2, batch_size=N, epochs=2000
)


# Check on the model ------------------------------------------------------


get_weights(model)
coef(sf1)


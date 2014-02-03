# This code runs a simple logistic regression on 3 variables of the training set and makes predictions for the test set. 
# THen a submission file is generated that contains the prediciton.

rm(list=ls())

start.time=proc.time()[3]  # Starts a clock to measure run time

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_08.R")

setwd("/Users/bowenpan/Dropbox/Stanford/Academia/Year 2/Winter 2013/OIT367 - Big Data/Problem sets/Kaggle")    # UPDATE THIS TO THE FOLDER THAT INCLUDES COMPETITION FILES

training = read.csv("training.csv")

test = read.csv("test.csv")

################ Logistic Regression ################

model = buildModel(response = "accepted", traindata = training, predictors = c("Amount_Approved","New_Rate"), type = "classify", method = "logisticRegression")

probabilities = genPred(model, newdata = test, method = "logisticRegression")

submission = data.frame(Id=test$Id, Prediction = probabilities)

write.csv(submission, file = "benchmark.csv", row.names = FALSE)

end.time=proc.time()[3]  # records current time to calculate overall code's run-time

cat("This code took ", end.time-start.time, " seconds\n")

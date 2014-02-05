#--------------------------------------------------------------------------------------------#
# Step A: This step cleans the memory, reads in the data, separates the predictors and response in the data and stores them in different 
# variables.

rm(list=ls())

#start.time=proc.time()[3]  # Starts a clock to measure run time

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_08.R")

setwd("/Users/bowenpan/Dropbox/Stanford/Academia/Year 2/Winter 2013/OIT367 - Big Data/Problem sets/Kaggle/Rcode")    # UPDATE THIS TO THE FOLDER THAT INCLUDES COMPETITION FILES

data = read.csv("training.csv")

response = as.numeric(data$accepted == 1)

predictors = data[,!(colnames(data) %in% c("Id"
                                           , "accepted"
                                           , "weeks"
                                           , "months"
                                           , "CarType"))] # redundant predictor

#--------------------------------------------------------------------------------------------#
# Step B: The data set contains application and approval dates. Also get rid of funding dates since this data does not exist for our test set. 
#         This step converts them into numeric time durations.

predictors$daysFromApply <- as.numeric(as.Date("2003-12-31") - as.Date(data$Apply_Date, format ="%Y-%m-%d"))
predictors$daysFromApprove <- as.numeric(as.Date("2003-12-31") - as.Date(data$Approve_Date, format ="%Y-%m-%d"))
#predictors$daysFromFund <- as.numeric(as.Date("2003-12-31") - as.Date(data$Fund_Date, format ="%Y-%m-%d"))

predictors$Apply_Date <- NULL 
predictors$Approve_Date <- NULL 
predictors$Fund_Date <-NULL 

#--------------------------------------------------------------------------------------------#
# Step C:  Calculates the number of unique values taken by each variable. 
#          Identifies and separates the categorical and numerical variables.

names.categoricals = c("Tier"
                       , "State"
                       , "Type"
                       , "Term"
                       , "termclass"
                       , "partnerbin"
                       , "CarType_id")

names.numericals = c("Primary_FICO"
                     , "New_Rate"
                     , "Used_Rate"
                     , "Amount_Approved"
                     , "Previous_Rate"
                     , "Competition_rate"
                     , "rate"
                     , "onemonth"
                     , "days"
                     , "rate1"
                     , "rel_compet_rate"
                     , "mp"
                     , "mp_rto_amtfinance"
                     , "daysFromApply"
                     , "daysFromApprove")

categorical.predictors = predictors[,names.categoricals]
numerical.predictors = predictors[,names.numericals]

#--------------------------------------------------------------------------------------------#
# Step D:  Converts our categorical variables in categorical.predictors into dummy variables. 

categoricals.binned = bin.cat.data(categorical.predictors)


#--------------------------------------------------------------------------------------------#
# Step E: Puts binned categorical predictors, numerical predictors, and the response into a new data frame

newPredictors = cbind(categoricals.binned, numerical.predictors)
data.binned = as.data.frame(cbind(newPredictors,response))

#--------------------------------------------------------------------------------------------#
# Step F: Sort the data by recency and partition the test segment

order.by.dates <- order(data.binned$daysFromApply)
data.binned.sorted <- data.binned[order.by.dates,]

training.validation.binned <- data.binned.sorted[1:82764,]
our.test.binned <- data.binned.sorted[-(1:82764),]


#########################################################################################
# PART II

#--------------------------------------------------------------------------------------------#
# Step A: Randomize and partition the training and validation sets

training.validation.binned.randomized = training.validation.binned[sample(nrow(training.validation.binned)),]

training = training.validation.binned.randomized[1:52141,]
validation = training.validation.binned.randomized[-(1:52141),]

#--------------------------------------------------------------------------------------------#
# Step B: Define some predictors. Leave old models in comments. Use agrep() for categorical predictor names.


# Based on correlations
# predictor.names = c("Used_Rate", agrep("Tier",names(newPredictors),value=TRUE), "Competition_rate",
#                    agrep("termclass",names(newPredictors),value=TRUE,max.distance=1), "rel_compet_rate",
#                    "mp", "Previous_Rate", agrep("partnerbin",names(newPredictors),value=TRUE),
#                    agrep("CarType",names(newPredictors),value=TRUE))

# predictor.names = c('Amount_Approved', 'New_Rate', 'rel_compet_rate')

predictor.names = c('daysFromApply',
                    'rate1',
                    'partnerbin2',
                    'TypeR',
                    'CarType.id2',
                    'CarType.id3',
                    'rel_compet_rate',
                    'Amount_Approved',
                    'mp', 
                    'Primary_FICO')

#--------------------------------------------------------------------------------------------#
# Step C: Run some models

model.logreg = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'logisticRegression')
pred.logreg  = genPred(model.logreg, newdata=validation, method='logisticRegression')

model.tree = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'decisionTree')
pred.tree  = genPred(model.tree, newdata=validation, method='decisionTree')

logreg.auc = auc(validation$response, pred.logreg)
tree.auc = auc(validation$response, pred.tree)

cat("Validation AUC of tree = ", tree.auc )
cat("Validation AUC of logreg = ", logreg.auc )


#########################################################################################
# PART III
#RUN THE MODELS ON OUR TEST DATA

our.test.logreg = genPred(model.logreg, newdata=our.test.binned, method='logisticRegression')
our.test.tree  = genPred(model.tree, newdata=our.test.binned, method='decisionTree')

our.test.logreg.auc = auc(our.test.binned$response, our.test.logreg)
our.test.tree.auc = auc(our.test.binned$response, our.test.tree)

cat("Test AUC of tree = ", our.test.tree.auc )
cat("Test AUC of logreg = ", our.test.logreg.auc )


#########################################################################################
# PART IV
#CREATING BINNED DATA FOR SUBMISSION TEST

test = read.csv("test.csv")

#test = test[,!(colnames(test) %in% c("Id"
#                                     , "accepted"
#                                     , "weeks"
#                                     , "months"
#                                     , "CarType"))] # redundant predictor

#--------------------------------------------------------------------------------------------#
# Step B: The data set contains application, approval and funding dates.
#         This step converts them into numeric time durations.

test$daysFromApply <- as.numeric(as.Date("2004-12-31") - as.Date(test$Apply_Date, format ="%Y-%m-%d"))
test$daysFromApprove <- as.numeric(as.Date("2004-12-31") - as.Date(test$Approve_Date, format ="%Y-%m-%d"))

test$Apply_Date <- NULL 
test$Approve_Date <- NULL 

#--------------------------------------------------------------------------------------------#
# Step C:  Calculates the number of unique values taken by each variable. 
#          Identifies and separates the categorical and numerical variables.

names.categoricals = c("Tier"
                       , "State"
                       , "Type"
                       , "Term"
                       , "termclass"
                       , "partnerbin"
                       , "CarType_id")

names.numericals = c("Primary_FICO"
                     , "New_Rate"
                     , "Used_Rate"
                     , "Amount_Approved"
                     , "Previous_Rate"
                     , "Competition_rate"
                     , "rate"
                     , "onemonth"
                     , "days"
                     , "rate1"
                     , "rel_compet_rate"
                     , "mp"
                     , "mp_rto_amtfinance"
                     , "daysFromApply"
                     , "daysFromApprove")

categorical.predictors = test[,names.categoricals]
numerical.predictors = test[,names.numericals]

#--------------------------------------------------------------------------------------------#
# Step D:  Converts our categorical variables in categorical.predictors into dummy variables. 

categoricals.binned = bin.cat.data(categorical.predictors)


#--------------------------------------------------------------------------------------------#
# Step E: Puts binned categorical predictors, numerical predictors, and the response into a new data frame

test.binned = cbind(categoricals.binned, numerical.predictors)


#########################################################################################
# SUBMISSION

#--------------------------------------------------------------------------------------------#
# Step A: Choose the model to output

submission.model <- model.logreg # manually type the model

#--------------------------------------------------------------------------------------------#
# Step B: The data set contains application, approval and funding dates.
#         This step converts them into numeric time durations.

predictors$daysFromApply <- as.numeric(as.Date("2003-12-31") - as.Date(data$Apply_Date, format ="%Y-%m-%d"))
predictors$daysFromApprove <- as.numeric(as.Date("2003-12-31") - as.Date(data$Approve_Date, format ="%Y-%m-%d"))
#predictors$daysFromFund <- as.numeric(as.Date("2003-12-31") - as.Date(data$Fund_Date, format ="%Y-%m-%d"))
predictors$lagTime <- predictors$daysFromApply - predictors$daysFromApprove

predictors$Apply_Date <- NULL 
predictors$Approve_Date <- NULL 
predictors$Fund_Date <-NULL 

#--------------------------------------------------------------------------------------------#
# Step C:  Calculates the number of unique values taken by each variable. 
#          Identifies and separates the categorical and numerical variables.

names.categoricals = c("Tier"
                       , "State"
                       , "Type"
                       , "Term"
                       , "termclass"
                       , "partnerbin"
                       , "CarType_id")

names.numericals = c("Primary_FICO"
                     , "New_Rate"
                     , "Used_Rate"
                     , "Amount_Approved"
                     #, "Previous_Rate" -- If we want to use this, we have to bin it
                     , "Competition_rate"
                     , "rate"
                     , "onemonth"
                     , "days"
                     , "rate1"
                     , "rel_compet_rate"
                     , "mp"
                     , "mp_rto_amtfinance"
                     , "daysFromApply"
                     , "daysFromApprove"
                     , "lagTime")

categorical.predictors = predictors[,names.categoricals]
numerical.predictors = predictors[,names.numericals]

#--------------------------------------------------------------------------------------------#
# Step D:  Converts our categorical variables in categorical.predictors into dummy variables. 

categoricals.binned = bin.cat.data(categorical.predictors)


#--------------------------------------------------------------------------------------------#
# Step E: Puts binned categorical predictors, numerical predictors, and the response into a new data frame

newPredictors = cbind(categoricals.binned, numerical.predictors)
data.binned = as.data.frame(cbind(newPredictors,response))

#--------------------------------------------------------------------------------------------#
# Step F: Sort the data by recency and partition the test segment

order.by.dates <- order(data.binned$daysFromApply)
data.binned.sorted <- data.binned[order.by.dates,]

training.validation.binned <- data.binned.sorted[1:82764,]
our.test.binned <- data.binned.sorted[-(1:82764),]


#########################################################################################
# PART II

#--------------------------------------------------------------------------------------------#
# Step A: Randomize and partition the training and validation sets using cross-validation

training.validation.binned.randomized = training.validation.binned[sample(nrow(training.validation.binned)),]

# This 
#training = training.validation.binned.randomized[1:52141,]
#validation = training.validation.binned.randomized[-(1:52141),]

##------------------------------- CROSS VALIDATION

nfolds = 3

flds <- createFolds(seq(1, nrow(training.validation.binned.randomized)), nfolds, list = TRUE, returnTrain = FALSE)

logreg.aucs = rep(0,nfolds)
tree.aucs = rep(0,nfolds)
svm.aucs = rep(0,nfolds)
combined.aucs =  rep(0,nfolds)

for(i in 1:nfolds){
  ind = flds[[i]]
  training = training.validation.binned.randomized[-ind,]
  validation = training.validation.binned.randomized[ind,]
  
  #--------------------------------------------------------------------------------------------#
  # Step B: Define some predictors. Leave old models in comments. Use agrep() for categorical predictor names.
  
  
  # Calculate and print the correlations
  #correlations <- as.data.frame(cor(training.validation.binned.randomized))
  #top.correlations <- order(abs(correlations$response))
  #correlations.sorted <- correlations[top.correlations,]
  #tail(correlations.sorted['response'],8)
  
  # Based on correlations (Above)
  predictor.names = c('TypeR','CarType.id3','Competition_rate','Amount_Approved','mp','CarType.id2')
  
  
  # Based on correlations (old)
  #predictor.names = c("Used_Rate", agrep("Tier",names(newPredictors),value=TRUE), "Competition_rate",
  #                   agrep("termclass",names(newPredictors),value=TRUE,max.distance=1), "rel_compet_rate",
  #                   "mp", agrep("partnerbin",names(newPredictors),value=TRUE),
  #                   agrep("CarType",names(newPredictors),value=TRUE), "New_Rate", "Amount_Approved")
  #"Previous_Rate"
  
  # predictor.names = c("Used_Rate", "Competition_rate", "rel_compet_rate", "mp", "Previous_Rate",
  #  "New_Rate", "Amount_Approved")
  
  #predictor.names = c('Amount_Approved', 'New_Rate', 'Used_Rate', 'Competition_rate', 'mp', 'rel_compet_rate')
  
  # predictor.names = c(agrep("Tier",names(newPredictors),value=TRUE), 'Amount_Approved', 'rel_compet_rate')
  
  # predictor.names = c(agrep("Tier",names(newPredictors),value=TRUE), 'Amount_Approved', 'rel_compet_rate', 'lagTime')
  
  
  
  
  #--------------------------------------------------------------------------------------------#
  # Step C: Run some models
  
  model.logreg = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'logisticRegression')
  pred.logreg  = genPred(model.logreg, newdata=validation, method='logisticRegression')
  
  model.tree = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'decisionTree', prune.tree=TRUE)
  pred.tree  = genPred(model.tree, newdata=validation, method='decisionTree')
  
  model.svm = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'nonlinearSVM')
  pred.svm  = genPred(model.svm, newdata=validation, method='nonlinearSVM')
  
  combined.preds = (pred.logreg + pred.tree + pred.svm)/3
  
  logreg.aucs[i] = auc(validation$response, pred.logreg)
  tree.aucs[i] = auc(validation$response, pred.tree)
  svm.aucs[i] = auc(validation$response, pred.svm)
  
  combined.aucs[i] = auc(validation$response, combined.preds)
  
  cat("Finished fold ", i, " of ", nfolds, "\n" )
  
}

cat("Average AUC of tree = ", mean(tree.aucs), " with standard error = ", sd(tree.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of logreg = ", mean(logreg.aucs), " with standard error = ", sd(logreg.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of svm = ", mean(svm.aucs), " with standard error = ", sd(svm.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of combined model = ", mean(combined.aucs), " with standard error = ", sd(combined.aucs)/sqrt(nfolds), "\n" )


#########################################################################################
# PART III
#RUN THE MODELS ON OUR TEST DATA

our.test.logreg = genPred(model.logreg, newdata=our.test.binned, method='logisticRegression')
our.test.tree  = genPred(model.tree, newdata=our.test.binned, method='decisionTree')
our.test.svm  = genPred(model.svm, newdata=our.test.binned, method='nonlinearSVM')

our.test.logreg.auc = auc(our.test.binned$response, our.test.logreg)
our.test.tree.auc = auc(our.test.binned$response, our.test.tree)
our.test.svm.auc = auc(our.test.binned$response, our.test.svm)

cat("AUC of tree on test data = ", our.test.tree.auc )
cat("AUC of logreg on test data = ", our.test.logreg.auc )
cat("AUC of svm on test data = ", our.test.svm.auc )

#------------------------ Build ensemble ---------------------------------------------------#
# Ensemble model

ensemble.training = data.frame(logreg = pred.logreg, tree = pred.tree, svm = pred.svm, response = validation$response)
ensemble.test = data.frame(logreg = our.test.logreg, tree=  our.test.tree,  svm = our.test.svm)

model.ensemble = buildModel("response", predictors=c(), traindata=ensemble.training, type = "classify", method = "logisticRegression")
pred.ensemble.test = genPred(model.ensemble, newdata = ensemble.test , method = "logisticRegression")

auc.ensemble.test = auc(our.test.binned$response,pred.ensemble.test)

cat("AUC of ensemble on test data = ",auc.ensemble.test,"\n")


#########################################################################################
# PART IV
#CREATING BINNED DATA FOR SUBMISSION TEST

test = read.csv("test.csv")

#test = test[,!(colnames(test) %in% c("Id"
#                                     , "accepted"
#                                     , "weeks"
#                                     , "months"
#                                     , "CarType"))] # redundant predictor

#--------------------------------------------------------------------------------------------#
# Step B: The data set contains application, approval and funding dates.
#         This step converts them into numeric time durations.

test$daysFromApply <- as.numeric(as.Date("2004-12-31") - as.Date(test$Apply_Date, format ="%Y-%m-%d"))
test$daysFromApprove <- as.numeric(as.Date("2004-12-31") - as.Date(test$Approve_Date, format ="%Y-%m-%d"))
test$lagTime = test$daysFromApprove - test$daysFromApply

test$Apply_Date <- NULL 
test$Approve_Date <- NULL 

#--------------------------------------------------------------------------------------------#
# Step C:  Calculates the number of unique values taken by each variable. 
#          Identifies and separates the categorical and numerical variables.

names.categoricals = c("Tier"
                       , "State"
                       , "Type"
                       , "Term"
                       , "termclass"
                       , "partnerbin"
                       , "CarType_id")

names.numericals = c("Primary_FICO"
                     , "New_Rate"
                     , "Used_Rate"
                     , "Amount_Approved"
                     , "Previous_Rate"
                     , "Competition_rate"
                     , "rate"
                     , "onemonth"
                     , "days"
                     , "rate1"
                     , "rel_compet_rate"
                     , "mp"
                     , "mp_rto_amtfinance"
                     , "daysFromApply"
                     , "daysFromApprove"
                     , "lagTime")

categorical.predictors = test[,names.categoricals]
numerical.predictors = test[,names.numericals]

#--------------------------------------------------------------------------------------------#
# Step D:  Converts our categorical variables in categorical.predictors into dummy variables. 

categoricals.binned = bin.cat.data(categorical.predictors)


#--------------------------------------------------------------------------------------------#
# Step E: Puts binned categorical predictors, numerical predictors, and the response into a new data frame

test.binned = cbind(categoricals.binned, numerical.predictors)


#########################################################################################
# SUBMISSION

#--------------------------------------------------------------------------------------------#
# Step A: Choose the model to output

submission.model <- model.logreg # manually type the model

#--------------------------------------------------------------------------------------------#
# Step B: Generate and output the submission data

probabilities = genPred(submission.model, newdata = test.binned, method = "logisticRegression") # make sure it's the right method

submission = data.frame(Id=test$Id, Prediction = probabilities)

write.csv(submission, file = "SupportVectorCuisine.csv", row.names = FALSE)


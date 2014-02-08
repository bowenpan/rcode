#--------------------------------------------------------------------------------------------#
# Step A: This step cleans the memory, reads in the data, separates the predictors and response in the data and stores them in different 
# variables.

rm(list=ls())

#start.time=proc.time()[3]  # Starts a clock to measure run time

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_09.R")

setwd("/Users/bowenpan/Dropbox/Stanford/Academia/Year 2/Winter 2013/OIT367 - Big Data/Problem sets/Kaggle/Rcode") # UPDATE THIS TO THE FOLDER THAT INCLUDES COMPETITION FILES

data = read.csv("training.csv")

#removing outliers in the creditRatio column 
outlier_tf = outlier(data$Amount_Approved,logical=TRUE)
find_outlier = which(outlier_tf==TRUE,arr.ind=TRUE)
data = data[-find_outlier,]


response = as.numeric(data$accepted == 1)

predictors = data[,!(colnames(data) %in% c("Id"
                                           , "accepted"
                                           , "weeks"
                                           , "months"
                                           , "CarType"))] # redundant predictor

#--------------------------------------------------------------------------------------------#
# Step B: Creation of New Variables
#         The data set contains application, approval and funding dates.
#         This step converts them into numeric time durations.
#         It also creates a categorical for the application and approval day of week and month. and
#         computes "credit ratio" of loan amount/FICO score


predictors$daysFromApply <- as.numeric(as.Date("2004-12-31") - as.Date(data$Apply_Date, format ="%Y-%m-%d"))
predictors$daysFromApprove <- as.numeric(as.Date("2004-12-31") - as.Date(data$Approve_Date, format ="%Y-%m-%d"))
predictors$lagTime <- predictors$daysFromApply - predictors$daysFromApprove

predictors$applyDay <- weekdays(as.Date(data$Apply_Date, format ="%Y-%m-%d"))
predictors$approveDay <- weekdays(as.Date(data$Approve_Date, format ="%Y-%m-%d"))
predictors$applyMonth <- months(as.Date(data$Apply_Date, format ="%Y-%m-%d"))
predictors$approveMonth <- months(as.Date(data$Approve_Date, format ="%Y-%m-%d"))

predictors$creditRatio = predictors$Amount_Approved / predictors$Primary_FICO

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
                       , "CarType_id"
                       , "applyDay"
                       , "applyMonth"
                       , "approveDay"
                       , "approveMonth")

names.numericals = c("Primary_FICO"
                     , "New_Rate"
                     , "Used_Rate"
                     , "Amount_Approved"
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
                     , "lagTime"
                     , "creditRatio")

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

order.by.dates <- order(data.binned$daysFromApprove)
data.binned.sorted <- data.binned[order.by.dates,]

training.validation.binned <- data.binned.sorted[1:98528,]
our.test.binned <- data.binned.sorted[-(1:98528),]


#########################################################################################
# PART II

#--------------------------------------------------------------------------------------------#
# Step A: Randomize and partition the training and validation sets using cross-validation

training.validation.binned.randomized = training.validation.binned[sample(nrow(training.validation.binned)),]

##------------------------------- CROSS VALIDATION

nfolds = 3

flds <- createFolds(seq(1, nrow(training.validation.binned.randomized)), nfolds, list = TRUE, returnTrain = FALSE)

logreg.aucs = rep(0,nfolds)
tree.aucs = rep(0,nfolds)
nb.aucs = rep(0,nfolds)
combined.aucs =  rep(0,nfolds)

for(i in 1:nfolds){
  ind = flds[[i]]
  training = training.validation.binned.randomized[-ind,]
  validation = training.validation.binned.randomized[ind,]
  
  #--------------------------------------------------------------------------------------------#
  # Step B: Define some predictors. Leave old models in comments. Use agrep() for categorical predictor names.
  
  
  ### CURRENT PREDICTORS:
  
  predictor.names = c('rel_compet_rate','Amount_Approved','mp','creditRatio')
  
  #--------------------------------------------------------------------------------------------#
  # Step C: Run some models
  
  model.logreg = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'logisticRegression')
  pred.logreg  = genPred(model.logreg, newdata=validation, method='logisticRegression')
  
  model.tree = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'decisionTree', prune.tree=TRUE)
  pred.tree  = genPred(model.tree, newdata=validation, method='decisionTree')
  
  model.nb = buildModel("response", predictors=predictor.names, training, type ='classify', method = 'NB')
  pred.nb  = genPred(model.nb, newdata=validation, method='NB')
  
  combined.preds = (pred.logreg + pred.tree + pred.nb)/3
  
  logreg.aucs[i] = auc(validation$response, pred.logreg)
  tree.aucs[i] = auc(validation$response, pred.tree)
  nb.aucs[i] = auc(validation$response, pred.nb)
  
  combined.aucs[i] = auc(validation$response, combined.preds)
  
  cat("Finished fold ", i, " of ", nfolds, "\n" )
  
}

cat("Average AUC of tree = ", mean(tree.aucs), " with standard error = ", sd(tree.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of logreg = ", mean(logreg.aucs), " with standard error = ", sd(logreg.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of NB = ", mean(nb.aucs), " with standard error = ", sd(nb.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of combined model = ", mean(combined.aucs), " with standard error = ", sd(combined.aucs)/sqrt(nfolds), "\n" )


#########################################################################################
# PART III
#RUN THE MODELS ON OUR TEST DATA

our.test.logreg = genPred(model.logreg, newdata=our.test.binned, method='logisticRegression')
our.test.tree  = genPred(model.tree, newdata=our.test.binned, method='decisionTree')
our.test.nb  = genPred(model.nb, newdata=our.test.binned, method='NB')

our.test.logreg.auc = auc(our.test.binned$response, our.test.logreg)
our.test.tree.auc = auc(our.test.binned$response, our.test.tree)
our.test.nb.auc = auc(our.test.binned$response, our.test.nb)

cat("AUC of tree on test data = ", our.test.tree.auc )
cat("AUC of logreg on test data = ", our.test.logreg.auc )
cat("AUC of NB on test data = ", our.test.nb.auc )

#------------------------ Build ensemble ---------------------------------------------------#
# Ensemble model

# Use these if you want logreg, tree, and NB in the ensemble
ensemble.training = data.frame(logreg = pred.logreg, tree = pred.tree, nb = pred.nb, response = validation$response)
ensemble.our.test = data.frame(logreg = our.test.logreg, tree = our.test.tree, nb = our.test.nb)

model.ensemble = buildModel("response", predictors=c(), traindata=ensemble.training, type = "classify", method = "logisticRegression")
pred.ensemble.test = genPred(model.ensemble, newdata = ensemble.our.test , method = "logisticRegression")

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
#         It also creates a categorical for the application day of week and
#         computes "credit ratio" of loan amount/FICO score

test$daysFromApply <- as.numeric(as.Date("2004-12-31") - as.Date(test$Apply_Date, format ="%Y-%m-%d"))
test$daysFromApprove <- as.numeric(as.Date("2004-12-31") - as.Date(test$Approve_Date, format ="%Y-%m-%d"))
test$lagTime = test$daysFromApprove - test$daysFromApply

test$applyDay <- weekdays(as.Date(test$Apply_Date, format ="%Y-%m-%d"))
test$approveDay <- weekdays(as.Date(test$Approve_Date, format ="%Y-%m-%d"))
test$applyMonth <- months(as.Date(test$Apply_Date, format ="%Y-%m-%d"))
test$approveMonth <- months(as.Date(test$Approve_Date, format ="%Y-%m-%d"))

test$creditRatio = test$Amount_Approved / test$Primary_FICO
test$Apply_Date <- NULL 
test$Approve_Date <- NULL 

test.order.by.dates <- order(test$daysFromApprove)
test <- test[test.order.by.dates,]


#--------------------------------------------------------------------------------------------#
# Step C:  Calculates the number of unique values taken by each variable. 
#          Identifies and separates the categorical and numerical variables.

names.categoricals = c("Tier"
                       , "State"
                       , "Type"
                       , "Term"
                       , "termclass"
                       , "partnerbin"
                       , "CarType_id"
                       , "applyDay"
                       , "applyMonth"
                       , "approveDay"
                       , "approveMonth")

names.numericals = c("Primary_FICO"
                     , "New_Rate"
                     , "Used_Rate"
                     , "Amount_Approved"
                     #, "Previous_Rate"
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
                     , "lagTime"
                     , "creditRatio")

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

submission.model <- "model.ensemble" # manually type the model

#--------------------------------------------------------------------------------------------#
# Step B: Generate and output the submission data

if(submission.model == "model.ensemble")
{
  probs.logreg = genPred(model.logreg, newdata = test.binned, method = "logisticRegression")
  probs.tree = genPred(model.tree, newdata = test.binned, method = "decisionTree")
  probs.nb = genPred(model.nb, newdata = test.binned, method = "NB")
  ensemble.final = data.frame(logreg = probs.logreg, tree=  probs.tree, nb = probs.nb) 
  
  probabilities = genPred(model.ensemble, newdata = ensemble.final, method = "logisticRegression")
  
}else if(submission.model == "model.logreg"){
  probabilities = genPred(model.logreg, newdata = test.binned, method = "logisticRegression")
}else if(submission.model == "model.tree"){
  probabilities = genPred(model.tree, newdata = test.binned, method = "decisionTree")
}else if(submission.model == "model.nb"){
  probabilities = genPred(model.nb, newdata = test.binned, method = "NB")
} else {
  stop("Invalid submission model.")
}

submission = data.frame(Id=test$Id, Prediction = probabilities)

write.csv(submission, file = "SupportVectorCuisine.csv", row.names = FALSE)

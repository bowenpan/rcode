#--------------------------------------------------------------------------------------------#
# Step A: This step cleans the memory, reads in the data, separates the predictors and response in the data and stores them in different 
# variables.

rm(list=ls())

#start.time=proc.time()[3]  # Starts a clock to measure run time

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_08.R")

setwd("/Users/bowenpan/Dropbox/Stanford/Academia/Year 2/Winter 2013/OIT367 - Big Data/Problem sets/Kaggle")    # UPDATE THIS TO THE FOLDER THAT INCLUDES COMPETITION FILES

data = read.csv("training.csv")

response = as.numeric(data$accepted == 1)

predictors = data[,!(colnames(data) %in% c("Id"
                                           , "accepted"
                                           , "weeks"
                                           , "months"
                                           , "CarType"))] # redundant predictor

#--------------------------------------------------------------------------------------------#
# Step B: The data set contains application, approval and funding dates.
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

#########################################################################################
# PART II
#CREATING BINNED DATA FOR TEST

test = read.csv("test.csv")

test = test[,!(colnames(test) %in% c("Id"
                                           , "accepted"
                                           , "weeks"
                                           , "months"
                                           , "CarType"))] # redundant predictor

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
# PART III Model selection


ncols = ncol(data.binned)

predictor.names = names(data.binned)[1:ncols-1]

##------------------------------- CROSS VALIDATION

#pick how many folds you want. 
nfolds = 5

#from a package called caret. How many rows you want "nfolds", create 20 set of indices for the fold
flds <- createFolds(seq(1, nrow(data)), nfolds, list = TRUE, returnTrain = FALSE) 

logreg.aucs = rep(0,nfolds)
tree.aucs = rep(0,nfolds)
svm.aucs = rep(0,nfolds)
combined.aucs =  rep(0,nfolds)

for(i in 1:nfolds){
  ind = flds[[i]] #at every fold, give me the ith index
  cv.training = data.binned[-ind,] #give me everything except for the ith index for training (all other colors in the slides)
  cv.validation = data.binned[ind,] #give me everything for the ith index for validation (BLUE color in the slides)
  
  model.logreg = buildModel("response", predictors=predictor.names, cv.training, type ='classify', method = 'logisticRegression')
  pred.logreg  = genPred(model.logreg, newdata=cv.validation, method='logisticRegression')
  
  model.tree = buildModel("response", predictors=predictor.names, cv.training, type ='classify', method = 'decisionTree')
  pred.tree  = genPred(model.tree, newdata=cv.validation, method='decisionTree')
  
   model.svm = buildModel("response", predictors=predictor.names, cv.training, type ='classify', method = 'nonlinearSVM')
   pred.svm  = genPred(model.svm, newdata=cv.validation, method='nonlinearSVM')
   
   combined.preds = (pred.logreg + pred.tree + pred.svm)/3
   
  logreg.aucs[i] = auc(cv.validation$response, pred.logreg)
  tree.aucs[i] = auc(cv.validation$response, pred.tree)
   svm.aucs[i] = auc(cv.validation$response, pred.svm)
   
   combined.aucs[i] = auc(cv.validation$response, combined.preds)
  
  cat("Finished fold ", i, " of ", nfolds, "\n" )
}

cat("Average AUC of tree = ", mean(tree.aucs), " with standard error = ", sd(tree.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of logreg = ", mean(logreg.aucs), " with standard error = ", sd(logreg.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of svm = ", mean(svm.aucs), " with standard error = ", sd(svm.aucs)/sqrt(nfolds), "\n" )
cat("Average AUC of combined model = ", mean(combined.aucs), " with standard error = ", sd(combined.aucs)/sqrt(nfolds), "\n" )



# MODEL 2: 

rm(list=ls())
gc()

library(readr)
library(h2o)
library(caret)
library(doParallel)
library(parallel)


setwd('/home/q/Dropbox/insight/project/')
df = read_csv('finalData/DataPriorToModels_cut.csv')
df = data.frame(df)
dim(df)
###############################################################
# TRAIN AND TEST SPLITS
###############################################################

# order by date
df = df[order(df$date),]

#drop restaurant_id, prior_date, and open
non_numeric = c('restaurant_id', 'prior_date', 'open')
df = df[,-which(names(df) %in% c(non_numeric))]

#drop 'Var.1' variable
df = df[,-which(names(df) %in% 'Var.1')]

######
# MISSING DATA
######
missing = colnames(df)[apply(is.na(df), 2, any)]

#train_X$Price.Range

#for now, just remove missing columns
if (length(missing)!=0){
  df2 = df[,-which(names(df) %in% missing)]
  df = df2
}

# CREATE TRAIN AND TEST DATA
exclude = c('vio1', 'vio2', 'vio3')

train_X = df[df$date < as.Date('2014-01-01'),-which(names(df) %in% c(exclude))]
test_X = df[df$date >= as.Date('2014-01-01'),-which(names(df) %in% c(exclude))]

y1_train = df[df$date < as.Date('2014-01-01'),'vio1']
y2_train = df[df$date < as.Date('2014-01-01'),'vio2']
y3_train = df[df$date < as.Date('2014-01-01'),'vio3']

y1_test = df[df$date >= as.Date('2014-01-01'),'vio1']
y2_test = df[df$date >= as.Date('2014-01-01'),'vio2']
y3_test = df[df$date >= as.Date('2014-01-01'),'vio3']

# y3 dummies
y3_train_dum = as.factor(ifelse(y3_train >=1, 'Y', 'N'))
y3_test_dum = as.factor(ifelse(y3_test >=1, 'Y', 'N'))

# # train --> h2o
# train_X_y3dum = train_X
# train_X_y3dum$y3_train_dum = as.factor(y3_train_dum)
# 
# # test --> h2o
# test_X_y3dum = test_X
# test_X_y3dum$y3_train_dum = as.factor(y3_test_dum)






trControl = trainControl(method = "none",
                         verboseIter = FALSE,
                         returnData = TRUE,
                         returnResamp = "final",
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary,
                         selectionFunction = "best",
                         allowParallel = TRUE
                         )


mtryGrid <- expand.grid(mtry = ceiling(sqrt(ncol(train_X))))


#cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
#registerDoParallel(cluster)

#[,c(2:16,19:238)]
rf = train(x=train_X[,c(2:16,19:238)],
           y=y3_train_dum,
           method ='rf',
           tuneGrid = mtryGrid,
           trControl=trControl,
           ntree=100)

#stopCluster(cl)

preds = predict(rf, newdata = test_X, type = "prob")
preds_raw = as.factor(ifelse(preds$Y > 0.5,'Y','N'))
                
sum(preds_raw==y3_test_dum)/length(y3_test_dum)

head(cbind(preds_raw,y3_test_dum))

#variable importances
imp = varImp(rf)
head(imp)
plot(imp)
options(scipen = 999)

# ranked variable imporances
impo = cbind(rownames(imp$importance),imp$importance)
impo[order(impo$Overall, decreasing=T),]

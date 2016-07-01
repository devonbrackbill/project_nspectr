# BASELINE MODEL (WEAKEST MODEL)
# no restaurant_id dummies; just use the historical time series.

# MODEL 1: 

#rm(list=ls())
#gc()

library(readr)
library(h2o)
library(pROC)
library(grid)
library(ROCR)
library(caret)
library(dplyr)

setwd('../../')
source('analysis/models/helper_functions.R')

# LOAD DATA
all_data = read.csv('finalData/DataPriorToModels.csv', stringsAsFactors = F)
all_data = data.frame(all_data)

###############################################################
# GLOBALS
###############################################################
NTREES = 1000

###############################################################
# TRAIN AND TEST SPLITS
###############################################################

# order by date
all_data$date = as.Date(all_data$date)
all_data = all_data[order(all_data$date),]

#drop restaurant_id, prior_data, and open
non_numeric = c('restaurant_id', 'prior_date', 'open')
all_data = all_data[,-which(names(all_data) %in% c(non_numeric))]

non_numeric = names(all_data[!sapply(all_data, is.numeric)])
all_data$date = as.Date(all_data$date)


#for now, just remove missing columns
missing = colnames(all_data)[apply(is.na(all_data), 2, any)]

if (length(missing)!=0){
  all_data2 = all_data[,-which(names(all_data) %in% missing)]
  all_data = all_data2
}

exclude = c('vio1', 'vio2', 'vio3')


train_X = all_data[all_data$date < as.Date('2014-01-01'),-which(names(all_data) %in% c(exclude))]
test_X = all_data[all_data$date >= as.Date('2014-01-01'),-which(names(all_data) %in% c(exclude))]

y1_train = all_data[all_data$date < as.Date('2014-01-01'),'vio1']
y2_train = all_data[all_data$date < as.Date('2014-01-01'),'vio2']
y3_train = all_data[all_data$date < as.Date('2014-01-01'),'vio3']

y1_test = all_data[all_data$date >= as.Date('2014-01-01'),'vio1']
y2_test = all_data[all_data$date >= as.Date('2014-01-01'),'vio2']
y3_test = all_data[all_data$date >= as.Date('2014-01-01'),'vio3']

# y3 dummies
y3_train_dum = ifelse(y3_train >=1, 1, 0)
y3_test_dum = ifelse(y3_test >=1, 1, 0)

# train --> h2o
train_X_y3dum = train_X
train_X_y3dum$y3_train_dum = as.factor(y3_train_dum)

# test --> h2o
test_X_y3dum = test_X
test_X_y3dum$y3_train_dum = as.factor(y3_test_dum)

#########################
# INITIALIZE H2O SERVER
#########################

localH2O = h2o.init(max_mem_size = '8G',
                    nthreads = -1)



# correct names of 2 variables that mess up h2o
names(train_X_y3dum)[612] = 'cafE_text'
names(train_X_y3dum)[1424] = 'entreE_text'
names(test_X_y3dum)[612] = 'cafE_text'
names(test_X_y3dum)[1424] = 'entreE_text'


# hex to import data into Java VM
#5005:7096
train.hex = as.h2o(train_X_y3dum[,c(3:5002,5005:7096)], 'train.hex')
h2o.head(train.hex)

test.hex = as.h2o(test_X_y3dum[,c(3:5002,5005:7096)], 'test.hex')
h2o.head(test.hex)


sink('names6.txt')
names(train.hex)
sink()

############################################################################
### RANDOM FOREST MODEL BASELINE: NO RESTAURANT DUMMIES
############################################################################

which.cols = c(5001:5004)

rf.model_0 = h2o.randomForest(x = c(which.cols),
                              y = 7092,
                              training_frame = train.hex,
                              ntree = NTREES,
                              mtries = -1,
                              validation = test.hex)
sink('model_diagnostics/m_0_diagnostics.txt')
rf.model_0

# variable importance
var.imp_0 = data.frame(h2o.varimp(rf.model_0))
print('**************************************')
print('MODEL_0')
print('**************************************')
print('VARIABLE IMPORTANCE')
print(var.imp_0[1:30,])

# confusion matrix
print('CONFUSION MATRIX')
h2o.confusionMatrix(rf.model_0)

# predictions
preds_0 = as.data.frame(h2o.predict(object = rf.model_0, newdata = test.hex))
preds_raw_0 = preds_0$predict
preds_raw_0_2 = ifelse(preds_0$p1 > 0.5, 1, 0)

# accuracy
print('ACCURACY (second one is with p = 0.5')
sum(preds_raw_0==y3_test_dum)/length(y3_test_dum)
sum(preds_raw_0_2==y3_test_dum)/length(y3_test_dum)

#auc
print('AUC')
print(auc(y3_test_dum,preds_0$p1))

# diagnostic plot
png(file='model_diagnostics/m_0_roc1.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
plot(roc(y3_test_dum,preds_0$p1), xlim=c(1,0))
dev.off()

png(file='model_diagnostics/m_0_pred_dist.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
plot_pred_type_distribution(data.frame(pred=preds_0$p1,
                                       actual=y3_test_dum), 0.5)
dev.off()


png(file='model_diagnostics/m_0_roc2.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
roc <- calculate_roc(data.frame(pred=preds_0$p1,
                                actual=y3_test_dum), 1, 1, n = 100)
plot_roc(roc, 0.4, 1, 1)
dev.off()

#PRECISION-RECALL
png(file='model_diagnostics/m_0_prec-recall.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
pred2 = prediction(preds_0$p1,y3_test_dum)
RP.perf <- performance(pred2, "prec", "rec")
plot (RP.perf)
dev.off()

png(file='model_diagnostics/m_0_roc3.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
ROC.perf <- performance(pred2, "tpr", "fpr")
plot (ROC.perf)
lines(seq(0,1,0.001), seq(0,1,0.001), lty=2)
dev.off()

# F1
print('PRECISION')
precision <- posPredValue(as.factor(preds_raw_0_2), as.factor(y3_test_dum))
precision

print('RECALL')
recall <- sensitivity(as.factor(preds_raw_0_2), as.factor(y3_test_dum))
recall

print('F1')
F1 <- (2 * precision * recall) / (precision + recall)
F1

sink()
# save model
h2o.saveModel(rf.model_0, path = "h2o_models/m_0", force = FALSE)

#rf.model_0 = h2o.loadModel(path = 'h2o_models/m4/DRF_model_R_1465466450879_8/')


###########################################
# WRITE PREDICTIONS TO FILE FOR MODEL 4
# FOR EACH RESTAURANT IN 2014/2015 (I.E., THEIR LAST VISIT)
###########################################


###############################################
# LAGGED VIOLATIONS ONLY
###############################################

# note: this uses data within the training set,
# so it's cheating, which increases the accuracy
pred_base = ifelse(test_X_y3dum$vio3_lag>0, 1, 0)

sum(pred_base==y3_test_dum)/length(y3_test_dum)

auc(y3_test_dum,pred_base)

# must correctly create all_data_store within this file
temp = all_data_store[all_data_store$date > as.Date('2014-01-01'),
                c('date', 'restaurant_id', 'vio3', 'vio3_lag')]

temp2 = data.frame(temp %>%
  group_by(restaurant_id) %>% 
    slice(which.min(date)))

temp2$vio3_dum = ifelse(temp2$vio3 > 0, 1, 0)
temp2$vio3_lag_dum = ifelse(temp2$vio3_lag > 0, 1, 0)

with(temp2, sum(vio3_lag_dum == vio3_dum)/nrow(temp2))

auc(temp2$vio3_dum, temp2$vio3_lag_dum)


# re-do the lagged data for the test set, pulling only 
# the data from the most recent training set

train_lag = data.frame(
  all_data_store %>% 
    filter(date < as.Date('2014-01-01')) %>% 
    group_by(restaurant_id) %>% 
    slice(which.max(date)))

baseline_X = merge(
  temp2,
  train_lag[,c('restaurant_id', 'date',
               'vio3_lag')],
  all.x = T, by = 'restaurant_id')

baseline_X$vio3_dum = ifelse(baseline_X$vio3>0, 1, 0)
baseline_X$vio3_lag_dum = ifelse(baseline_X$vio3_lag.y>0, 1, 0)

with(baseline_X, sum(vio3_dum == vio3_lag_dum, na.rm=T)/nrow(baseline_X))

auc(baseline_X$vio3_dum,baseline_X$vio3_lag_dum)

bootstrapAccuracy2 = function(x, K = 10000,quant=c(0.025, 0.975)){
  
  set.seed(1234)
  
  N = nrow(x)
  
  samp_mat = sample(1:nrow(x), K*N, replace=T)
  
  pred_mat = matrix(x$vio3_lag_dum[samp_mat], ncol=K)
  actual_mat = matrix(x$vio3_dum[samp_mat], ncol=K)
  
  out = rep(0,K)
  for (i in 1:ncol(pred_mat)){
    out[i] = sum(pred_mat[,i] == actual_mat[,i], na.rm=T)/N
  }
  quantile(out,c(quant))
}

bootstrapAccuracy2(baseline_X)

bootstrapAUC2 = function(x, K = 10000,quant=c(0.025, 0.975)){
  
  set.seed(1234)
  
  N = nrow(x)
  
  samp_mat = sample(1:nrow(x), K*N, replace=T)
  
  pred_mat = matrix(x$vio3_lag_dum[samp_mat], ncol=K)
  actual_mat = matrix(x$vio3_dum[samp_mat], ncol=K)
  
  out = rep(0,K)
  for (i in 1:ncol(pred_mat)){
    out[i] = auc(actual_mat[,i],pred_mat[,i])
  }
  quantile(out,c(quant))
}

bootstrapAUC2(baseline_X)


###### PREDICTIONS_5

predictions_5_first = data.frame(
  predictions_5 %>%
    group_by(restaurant_id) %>% 
    slice(which.min(date)))
    

sum(predictions_5_first$pred_raw == 
  predictions_5_first$vio_raw)/nrow(predictions_5_first)

auc(predictions_5_first$vio_raw, predictions_5_first$pred_raw)

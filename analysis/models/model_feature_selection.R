# MODEL RECURSIVE FEATURE ELIMINATION

# MODEL 1: 

rm(list=ls())

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
NTREES = 100

###############################################################
# TRAIN AND TEST SPLITS
###############################################################

# order by date
all_data$date = as.Date(all_data$date)
all_data = all_data[order(all_data$date),]
all_data_store = all_data

#drop restaurant_id, prior_data, and open
non_numeric = c('restaurant_id', 'prior_date', 'open')
all_data = all_data[,-which(names(all_data) %in% c(non_numeric))]

non_numeric = names(all_data[!sapply(all_data, is.numeric)])

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

# correct names of 2 variables that mess up h2o
names(train_X_y3dum)[612] = 'cafE_text'
names(train_X_y3dum)[1424] = 'entreE_text'
names(test_X_y3dum)[612] = 'cafE_text'
names(test_X_y3dum)[1424] = 'entreE_text'


#########################
# INITIALIZE H2O SERVER
#########################

localH2O = h2o.init(max_mem_size = '10G',
                    nthreads = -1)
 
# hex to import data into Java VM
train.hex = as.h2o(train_X_y3dum[,c(3:5002,5005:7096)], 'train.hex')
h2o.head(train.hex)

############################################################################
### RANDOM FOREST MODEL : RECURSIVE FEATURE ELIMINATION
############################################################################


### CROSS VALIDATION SETUP ###
# have to be careful to use time series cross-validation here,
# not random sampling among all data b/c we want the same structure
# as our train --> test data
# indexes for each train and test set
# partitioned by year (5-fold CV)

# 1. 2007 + 2008 > 2009
# 2. 2007 + 2008 + 2009 > 2010
# 3. 2007 + 2008 + 2009 + 2010 > 2011
# 4. 2007 + 2008 + 2009 > 2010 + 2011 > 2012
# 5. 2007 + 2008 + 2009 > 2010 + 2011 + 2012 > 2013

train_idx1 = which(train_X_y3dum$year < 2009)
test_idx1 = which(train_X_y3dum$year == 2009)
train_idx2 = which(train_X_y3dum$year < 2010)
test_idx2 = which(train_X_y3dum$year == 2010)
train_idx3 = which(train_X_y3dum$year < 2011)
test_idx3 = which(train_X_y3dum$year == 2011)
train_idx4 = which(train_X_y3dum$year < 2012)
test_idx4 = which(train_X_y3dum$year == 2012)
train_idx5 = which(train_X_y3dum$year < 2013)
test_idx5 = which(train_X_y3dum$year == 2013)

train_idx = list(train_idx1,train_idx2,train_idx3,train_idx4,
                 train_idx5)
test_idx = list(test_idx1,test_idx2,test_idx3,test_idx4,
                test_idx5)

# number of variables to keep (start with full model, and recursively eliminate)
num_keep_list = rev(c(50,100,200,500,1000,3000,ncol(train.hex)))
#num_keep_list = rev(c(5,10,20,30,40))

# storage is the dataframe of the results of this cross-validation
storage = data.frame(
  num_keep = rep(num_keep_list,5), #5 = num folds
  fold = rep(1:5,each=length(num_keep_list)),
  accuracy = 0L,
  auc= 0L,
  precision = 0L,
  recall = 0L,
  f1 =0L)

counter = 1
# for loop for each CV fold
for (i in 4:length(train_idx)){
  
  include = c(1:7091,7092) # start with all variables
  
  # for loop for each number of variables to include
  for (num_keep in num_keep_list){
    print(paste0(counter, '/', nrow(storage),
                 ' = ', round(100*counter/nrow(storage),0), '%'))
    
    if (num_keep != ncol(train.hex)){
      
      # remove everything except the top num_keep variables
      include = which(names(train.hex) %in% 
                        c(var.imp_cv$variable[1:num_keep]))
      print(paste0(length(include), ' columns'))
    }

    rf.model_cv = h2o.randomForest(
      x = include,
      y = 7092,
      training_frame = train.hex[train_idx[[i]],1:7092],
      ntree = NTREES,
      mtries = -1,
      validation = train.hex[test_idx[[i]],1:7092])
    
    # variable importance
    var.imp_cv = data.frame(h2o.varimp(rf.model_cv))
    var.imp_cv = var.imp_cv[order(var.imp_cv$scaled_importance,
                                  decreasing=T),]
    
    # predictions
    preds_cv = as.data.frame(h2o.predict(
      object = rf.model_cv, 
      newdata = train.hex[test_idx[[i]],1:7092]))
    
    preds_raw_cv = preds_cv$predict
    preds_raw_cv_2 = ifelse(preds_cv$p1 > 0.5, 1, 0)
    
    # accuracy
    
    my_acc = sum(preds_raw_cv_2==
                   y3_train_dum[test_idx[[i]] ])/length(y3_test_dum)
    
    #auc
    my_auc = auc(y3_train_dum[test_idx[[i]] ],preds_cv$p1)
    
    # F1
    precision <- posPredValue(
      as.factor(preds_raw_cv_2), 
      as.factor(y3_train_dum[test_idx[[i]] ]))
    recall <- sensitivity(as.factor(preds_raw_cv_2),
                          as.factor(y3_train_dum[test_idx[[i]] ]))
    F1 <- (2 * precision * recall) / (precision + recall)
    
    
    storage[counter,] = c(num_keep, i, my_acc,
                          my_auc, precision, recall,
                          F1)
    counter = counter + 1
    gc()
  }
}

# write.csv(storage,file='finalData/recursiveFeatureElim-10trees.csv',
#           row.names = F)
# write.csv(storage,
#           file='finalData/recursiveFeatureElim-10trees-lowRange.csv',
#           row.names = F)

head(storage)

# plots for each 
storage2 = data.frame(storage %>% 
                        filter(fold != 4) %>% 
                        group_by(num_keep) %>% 
                        summarize(
                          accuracy.mean = mean(accuracy),
                          auc.mean = mean(auc),
                          precision.mean = mean(precision),
                          recall.mean = mean(recall),
                          f1.mean = mean(f1),
                          accuracy.sd = sd(accuracy),
                          auc.sd = sd(auc),
                          precision.sd = sd(precision),
                          recall.sd = sd(recall),
                          f1.sd = sd(f1)
                        ))
storage2

my_plot = function(x, y.var, ylim=c(.5,1)){
  
  y.var.mean = paste0(y.var,'.mean')
  y.var.sd = paste0(y.var,'.sd')
  
  plot(x[,y.var.mean]~x$num_keep, ylim=ylim,
       type='l', xlab='Number of Variables',
       ylab=y.var, las=1)
  arrows(x$num_keep,
    x[,y.var.mean]-1.96*x[,y.var.sd],
    x$num_keep,
    x[,y.var.mean]+1.96*x[,y.var.sd],
    code = 0)
}

# the confidence intervals are not tight enough to distinguish among the lower ranges,
# so we choose 200 variables as a reasonable number of features to keep.
my_plot(storage2,'accuracy',c(.65,.8))
my_plot(storage2,'auc',ylim=c(.65,.85))
my_plot(storage2,'precision',c(.7,.8))
my_plot(storage2,'recall', c(.7,.9))
my_plot(storage2,'f1',c(.7,.82))


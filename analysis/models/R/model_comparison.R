# MODEL 2 & 4/5 COMPARISON

library(pROC)
library(caret)
library(DMwR)

setwd('C:/Users/Devon/Dropbox/insight/project')
if (Sys.info()[['sysname']] == 'Linux') setwd('/home/q/Dropbox/insight/project')
predictions_2 = read.csv('finalData/predictions_2.csv')
predictions_5 = read.csv('finalData/predictions_final.csv')

### CLEAN DATA ###
cleaner = function(x){
  x$restaurant_id = as.character(x$restaurant_id)
  x$date = as.Date(as.character(x$date))
  return(x)
}

predictions_2 = cleaner(predictions_2)
predictions_5 = cleaner(predictions_5)

### ACCURACY ###
acc = function(x){
  sum(x$pred_raw == x$vio_raw)/nrow(x)
}
acc(predictions_2)
acc(predictions_5)

### BOOTSTRAP ACCURACY CI'S ###

bootstrapAccuracy = function(x, K = 10000,quant=c(0.025, 0.975)){
  
  set.seed(1234)
  
  N = nrow(x)

  samp_mat = sample(1:nrow(x), K*N, replace=T)
  
  pred_mat = matrix(x$pred_raw[samp_mat], ncol=K)
  actual_mat = matrix(x$vio_raw[samp_mat], ncol=K)
  
  out = rep(0,K)
  for (i in 1:ncol(pred_mat)){
    out[i] = sum(pred_mat[,i] == actual_mat[,i])/N
  }
  quantile(out,c(quant))
}

bootstrapAccuracy(predictions_5, quant = c(0.05, 0.95))
bootstrapAccuracy(predictions_2, quant = c(0.05, 0.95))

### PAIRED COMPARISON ON ACCURACY ###

bootstrapAccuracyDistrib = function(x, K = 10000,quant=c(0.025, 0.975)){
  
  set.seed(1234)
  
  N = nrow(x)
  
  samp_mat = sample(1:nrow(x), K*N, replace=T)
  
  pred_mat = matrix(x$pred_raw[samp_mat], ncol=K)
  actual_mat = matrix(x$vio_raw[samp_mat], ncol=K)
  
  out = rep(0,K)
  for (i in 1:ncol(pred_mat)){
    out[i] = sum(pred_mat[,i] == actual_mat[,i])/N
  }
  out
}

distrib_2 = bootstrapAccuracyDistrib(predictions_2)
distrib_5 = bootstrapAccuracyDistrib(predictions_5)

wilcox.test(distrib_2, distrib_5, paired = T)
sum(distrib_5 > distrib_2)/length(distrib_5)


### ROC CURVES ###

auc(y3_test_dum,preds_2$p1)

my_auc = function(x){
  auc(x$vio_raw, x$pred)
}

my_auc(predictions_2)
my_auc(predictions_5)

bootstrapAUC = function(x, K = 10000,quant=c(0.025, 0.975), verbose=F){
  # bootstraps the AUC by repeatedly sampling from the testing set
  set.seed(1234)
  
  N = nrow(x)
  
  samp_mat = sample(1:nrow(x), K*N, replace=T)
  
  pred_mat = matrix(x$pred[samp_mat], ncol=K)
  actual_mat = matrix(x$vio_raw[samp_mat], ncol=K)
  
  out = rep(0,K)
  for (i in 1:ncol(pred_mat)){
    out[i] = auc(actual_mat[,i], pred_mat[,i])
    if (verbose == T){
      if (i %% 10 == 0) print(paste0(i,'/',K))
    }
  }
  quantile(out,c(quant))
}


bootstrapAUC(predictions_2, K = 100)
bootstrapAUC(predictions_5, K = 100)

my_auc(predictions_2)
my_auc(predictions_5)

### PRECISION, RECALL, AND F1 SCORES ###


f1Family = function(x, to_return='f1'){
  
  precision = posPredValue(as.factor(x$pred_raw),
                           as.factor(x$vio_raw))
  
  if (to_return == 'precision') return(precision)
  
  recall = sensitivity(as.factor(x$pred_raw),
                       as.factor(x$vio_raw))
  
  if (to_return == 'recall') return(recall)
  
  f1 = (2 * precision * recall) / (precision + recall)
  
  if (to_return == 'f1') return(f1)
  
}

f1Family(predictions_2)
f1Family(predictions_5)


f1Bootstrap = function(x, K = 10000,quant=c(0.025, 0.975), verbose=F,
                       to_return = 'f1'){
  # bootstraps the F1 by repeatedly sampling from the testing set
  set.seed(1234)
  
  N = nrow(x)
  
  samp_mat = sample(1:nrow(x), K*N, replace=T)
  
  pred_mat = matrix(x$pred_raw[samp_mat], ncol=K)
  actual_mat = matrix(x$vio_raw[samp_mat], ncol=K)
  
  out = rep(0,K)
  for (i in 1:ncol(pred_mat)){
    
    ### to do: calc all metrics and store here
    precision = posPredValue(as.factor(x$pred_raw),
                             as.factor(x$vio_raw))
    
    recall = sensitivity(as.factor(x$pred_raw),
                         as.factor(x$vio_raw))
    
    out[i] = (2 * precision * recall) / (precision + recall)
    
    if (verbose == T){
      if (i %% 10 == 0) print(paste0(i,'/',K))
    }
  }
  if (to_return == 'f1'){
    return(quantile(out,c(quant)))
  } else {
    return(out)
  }
}

f1Bootstrap(predictions_2, ver)
f1Bootstrap(predictions_5)

### PRECISION - RECALL CURVE ###
PRcurve(predictions_2$pred, predictions_2$vio_raw)
PRcurve(predictions_5$pred, predictions_5$vio_raw)



### PLOT ROC CURVES FOR BOTH MODELS ###

### VARIABLE IMPORTANCE ###
# (copied from m_5_diagnostics)

var.imp = read.csv('finalData/variable_importance.csv')

var.imp$variable2 = c('Violation (severe) Lag',
                     'Month',
                     'Time Since Last Inspection',
                     'Year',
                     'Violation (minor) Lag',
                     'Longitude',
                     'Latitude',
                     'Mean Review Score (filtered)',
                     'Mean Review Score (raw)',
                     '"place"',
                     '"food"',
                     '"good"',
                     'Review SD (raw)',
                     'Stars',
                     'Number of Reviews',
                     'Review SD (filtered)',
                     '"just"',
                     'Violation (medium) Lag',
                     'Noise Level',
                     '"like"',
                     '"great"',
                     '"really"',
                     'Delivery',
                     "don't",
                     "Wi-fi",
                     '"time"',
                     '"have"',
                     '"service"',
                     'Alcohol',
                     '"Boston"')

all_data = read.csv('finalData/DataPriorToModels.csv', stringsAsFactors = F)
all_data = data.frame(all_data)

# order by date
all_data$date = as.Date(all_data$date)
all_data = all_data[order(all_data$date),]

#for now, just remove missing columns
missing = colnames(all_data)[apply(is.na(all_data), 2, any)]

if (length(missing)!=0){
  all_data2 = all_data[,-which(names(all_data) %in% missing)]
  all_data = all_data2
}


train_X2 = all_data[all_data$date < as.Date('2014-01-01'),]

# y3 dummies
y3_train_dum = ifelse(y3_train >=1, 1, 0)
#y3_test_dum = ifelse(y3_test >=1, 1, 0)

# CORRELATIONS AMONG TOP 30 VARIABLES
var.names = as.character(var.imp$variable)
train_sub = train_X2[,c(var.names,'vio3')]
train_sub$vio3_dum = ifelse(train_sub$vio3>0,1,0)

cor_mat = data.frame(cor(train_sub))
cors = data.frame(cor=cor_mat[,'vio3_dum'],
                  var.name=row.names(cor_mat))
cors$color = ifelse(cors$cor > 0, 'lightgreen', 'red')
cors$var.name = as.character(cors$var.name)
var.imp$variable = as.character(var.imp$variable)

var.imp2 = merge(var.imp,cors, by.x = 'variable', 
                 by.y = 'var.name', all.x = T)
var.imp2 = var.imp2[order(var.imp2$scaled_importance, decreasing = T),]

# color all of these based on whether they increase or decrease violations

png('model_diagnostics/variableImportance-wide.png',
    height = 7, width = 7, units='in', type='cairo',
    res = 200)
par(mar=c(3,13,1,1))
barplot(rev(var.imp2$scaled_importance),
  horiz=T,
  names.arg = rev(var.imp2$variable2),
  las = 1,
  col = rev(var.imp2$color), border = NA)
legend('bottomright',
       c('Positive', 'Negative'),
       fill = c('lightgreen', 'red'),
       bty='n',
       title = 'Correlation')
dev.off()


png('model_diagnostics/variableImportanceMinus1-wide.png',
    height = 7, width = 7, units='in', type='cairo',
    res = 200)
par(mar=c(3,13,1,1))
barplot(rev(var.imp2$scaled_importance[-1]),
        horiz=T,
        names.arg = rev(var.imp2$variable2[-1]),
        las = 1,
        col = rev(var.imp2$color[-1]), border = NA)
legend('bottomright',
       c('Positive', 'Negative'),
       fill = c('lightgreen', 'red'),
       bty='n',
       title = 'Correlation')
dev.off()


# CORRELATION PLOT
library(corrplot)

par(mar = c(2,2,6,2))
png('model_diagnostics/corrPlot.png',
    height = 7, width = 7, units='in', type='cairo',
    res = 200)

corrplot(cor(train_sub), type='upper')
dev.off()
# MODEL 4: 

rm(list=ls())
gc()

library(readr)
library(h2o)
library(pROC)
library(grid)
library(ROCR)
library(caret)

#http://www.r-bloggers.com/illustrated-guide-to-roc-and-auc/

plot_pred_type_distribution <- function(df, threshold) {
  v <- rep(NA, nrow(df))
  v <- ifelse(df$pred >= threshold & df$actual == 1, "TP", v)
  v <- ifelse(df$pred >= threshold & df$actual == 0, "FP", v)
  v <- ifelse(df$pred < threshold & df$actual == 1, "FN", v)
  v <- ifelse(df$pred < threshold & df$actual == 0, "TN", v)
  
  df$pred_type <- v
  
  ggplot(data=df, aes(x=actual, y=pred)) + 
    geom_violin(fill=rgb(1,1,1,alpha=0.6), color=NA) + 
    geom_jitter(aes(color=pred_type), alpha=0.6) +
    geom_hline(yintercept=threshold, color="red", alpha=0.6) +
    scale_color_discrete(name = "type") +
    labs(title=sprintf("Threshold at %.2f", threshold))
}

calculate_roc <- function(df, cost_of_fp, cost_of_fn, n=100) {
  tpr <- function(df, threshold) {
    sum(df$pred >= threshold & df$actual == 1) / sum(df$actual == 1)
  }
  
  fpr <- function(df, threshold) {
    sum(df$pred >= threshold & df$actual == 0) / sum(df$actual == 0)
  }
  
  cost <- function(df, threshold, cost_of_fp, cost_of_fn) {
    sum(df$pred >= threshold & df$actual == 0) * cost_of_fp + 
      sum(df$pred < threshold & df$actual == 1) * cost_of_fn
  }
  
  roc <- data.frame(threshold = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tpr <- sapply(roc$threshold, function(th) tpr(df, th))
  roc$fpr <- sapply(roc$threshold, function(th) fpr(df, th))
  roc$cost <- sapply(roc$threshold, function(th) cost(df, th, cost_of_fp, cost_of_fn))
  
  return(roc)
}

plot_roc <- function(roc, threshold, cost_of_fp, cost_of_fn) {
  library(gridExtra)
  
  norm_vec <- function(v) (v - min(v))/diff(range(v))
  
  idx_threshold = which.min(abs(roc$threshold-threshold))
  
  col_ramp <- colorRampPalette(c("green","orange","red","black"))(100)
  col_by_cost <- col_ramp[ceiling(norm_vec(roc$cost)*99)+1]
  p_roc <- ggplot(roc, aes(fpr,tpr)) + 
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    coord_fixed() +
    geom_line(aes(threshold,threshold), color=rgb(0,0,1,alpha=0.5)) +
    labs(title = sprintf("ROC")) + xlab("FPR") + ylab("TPR") +
    geom_hline(yintercept=roc[idx_threshold,"tpr"], alpha=0.5, linetype="dashed") +
    geom_vline(xintercept=roc[idx_threshold,"fpr"], alpha=0.5, linetype="dashed")
  
  p_cost <- ggplot(roc, aes(threshold, cost)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    labs(title = sprintf("cost function")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")
  
  sub_title <- sprintf("threshold at %.2f - cost of FP = %d, cost of FN = %d", threshold, cost_of_fp, cost_of_fn)
  
  grid.arrange(p_roc, p_cost, ncol=2, sub=textGrob(sub_title, gp=gpar(cex=1), just="bottom"))
}

setwd('/home/q/Dropbox/insight/project/')
all_data = read_csv('finalData/DataPriorToModels_cut.csv')
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
all_data_store = all_data

sink('names3.txt')
names(all_data)
sink()

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

#########################
# INITIALIZE H2O SERVER
#########################

localH2O = h2o.init(max_mem_size = '8G',
                    nthreads = -1)

sink('names5.txt')
names(train_X_y3dum)
sink()



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

# RANDOM FOREST MODEL -----------------------------------------------------

############################################################################
### RANDOM FOREST MODEL 4: THRESHOLD FEATURE SELECTION
############################################################################

# this will need to be changed!!!!!
#which.cols = which(names(train.hex) %in% var.imp$variable[1:200])
which.cols = c(84,148,200,221,265,266,277,310,361,386,397,407,418,481,515,589,624,725,744,764,765,
               834,855,962,1000,1076,1106,1125,1164,1174,1214,1256,1311,1317,1357,1390,1466,1532,1547,1583,1632,1736,
               1750,1755,1761,1767,1772,1865,1871,1950,1978,2213,2232,2248,2363,2423,2453,2519,2544,2558,2575,2597,2602,
               2615,2623,2632,2652,2684,2691,2733,2739,2778,2822,2927,2942,2966,2967,3042,3046,3067,3170,3183,3231,3239,
               3306,3404,3429,3445,3516,3564,3607,3661,3698,3707,3731,3778,3797,3815,3829,3898,3921,4060,4118,4158,4189,
               4338,4348,4366,4410,4441,4455,4458,4462,4493,4513,4591,4611,4679,4695,4755,4794,4812,4837,4864,4930,4941,
               5001,5002,5003,5004,5005,5006,5007,5008,5009,5010,5011,5012,5013,5014,5015,5016,5017,5018,5019,5020,5021,
               5022,5023,5025,5026,5027,5028,5030,5032,5033,5036,5038,5039,5040,5041,5042,5043,5044,5047,5050,5051,5052,
               5053,5054,5055,5056,5057,5058,5059,5060,5061,5062,5063,5066,5069,5070,5071,5072,5073,5074,5086,5102,5113,
               5116,5141,5144,5171,5172,5199,5206,5221,5235,5239,5240)

rf.model_5 = h2o.randomForest(x = c(which.cols),
                              y = 7092,
                              training_frame = train.hex,
                              #classification = T,
                              ntree = NTREES,
                              #depth = 3000,
                              mtries = -1,
                              #nfolds = 3,
                              validation = test.hex)
sink('model_diagnostics/m_5_diagnostics.txt')
rf.model_5

# variable importance
var.imp_5 = data.frame(h2o.varimp(rf.model_5))
print('**************************************')
print('MODEL_5')
print('**************************************')
print('VARIABLE IMPORTANCE')
print(var.imp_5[1:30,])

# confusion matrix
print('CONFUSION MATRIX')
h2o.confusionMatrix(rf.model_5)

# predictions
preds_5 = as.data.frame(h2o.predict(object = rf.model_5, newdata = test.hex))
preds_raw_5 = preds_5$predict
preds_raw_5_2 = ifelse(preds_5$p1 > 0.5, 1, 0)

# accuracy
print('ACCURACY (second one is with p = 0.5')
sum(preds_raw_5==y3_test_dum)/length(y3_test_dum)
sum(preds_raw_5_2==y3_test_dum)/length(y3_test_dum)

#auc
print('AUC')
print(auc(y3_test_dum,preds_5$p1))

# diagnostic plot
png(file='model_diagnostics/m_5_roc1.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
plot(roc(y3_test_dum,preds_5$p1), xlim=c(1,0))
dev.off()

png(file='model_diagnostics/m_5_pred_dist.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
plot_pred_type_distribution(data.frame(pred=preds_5$p1,
                                       actual=y3_test_dum), 0.5)
dev.off()


png(file='model_diagnostics/m_5_roc2.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
roc <- calculate_roc(data.frame(pred=preds_5$p1,
                                actual=y3_test_dum), 1, 1, n = 100)
plot_roc(roc, 0.4, 1, 1)
dev.off()

#PRECISION-RECALL
png(file='model_diagnostics/m_5_prec-recall.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
pred2 = prediction(preds_5$p1,y3_test_dum)
RP.perf <- performance(pred2, "prec", "rec")
plot (RP.perf)
dev.off()

png(file='model_diagnostics/m_5_roc3.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
ROC.perf <- performance(pred2, "tpr", "fpr")
plot (ROC.perf)
lines(seq(0,1,0.001), seq(0,1,0.001), lty=2)
dev.off()

# F1
print('PRECISION')
precision <- posPredValue(as.factor(preds_raw_5_2), as.factor(y3_test_dum))
precision

print('RECALL')
recall <- sensitivity(as.factor(preds_raw_5_2), as.factor(y3_test_dum))
recall

print('F1')
F1 <- (2 * precision * recall) / (precision + recall)
F1

sink()
# save model
h2o.saveModel(rf.model_5, path = "h2o_models/m_5", force = FALSE)

#rf.model_5 = h2o.loadModel(path = 'h2o_models/m4/DRF_model_R_1465466450879_8/')


###########################################
# WRITE PREDICTIONS TO FILE FOR MODEL 4
# FOR EACH RESTAURANT IN 2014/2015 (I.E., THEIR LAST VISIT)
###########################################

predictions_5 = cbind.data.frame(
    all_data_store[all_data_store$date >=
                     as.Date('2014-01-01'),
                   c('restaurant_id', 'date',
                      'latitude', 'longitude', 'vio3')],
    preds_5$p1)
names(predictions_5)[length(names(predictions_5))] = 'pred'
predictions_5$pred_raw = ifelse(predictions_5$pred > 0.5, 1, 0)
predictions_5$vio_raw = ifelse(predictions_5$vio3 >= 1, 1, 0)

sum(predictions_5$pred_raw ==
      predictions_5$vio_raw)/nrow(predictions_5)
    
write.csv(predictions_5,'finalData/predictions_final.csv',
          row.names = F)


# to add in restaurant names
df = data.frame(read_csv('finalData/train_labels_no_null_with_business_info.csv'))
df$date = as.Date(df$date)


df2 = data.frame(df %>%
                   filter(date >= as.Date('2014-01-01')) %>% 
                   group_by(restaurant_id) %>%
                   slice(which.max(date)))

df2 = df2[,c('name', 'restaurant_id', 'date')]

df3 = merge(predictions_5, df2, all.x=T, all.y=F,
            by = c('restaurant_id','date'))




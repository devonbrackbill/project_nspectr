# LOGISTIC REGRESSION MODEL


mod_log = glm( family='binomial',data=train_X_y3dum)


rf.model_2 = h2o.randomForest(x = c(5001:5004,5239:7091),
                              y = 7092,
                              training_frame = train.hex,
                              #classification = T,
                              ntree = NTREES,
                              #depth = 3000,
                              mtries = -1,
                              #nfolds = 3,
                              validation = test.hex)
sink('model_diagnostics/m_2_diagnostics.txt')
rf.model_2

# variable importance
var.imp_2 = data.frame(h2o.varimp(rf.model_2))
print('**************************************')
print('MODEL_2')
print('**************************************')
print('VARIABLE IMPORTANCE')
print(var.imp_2[1:30,])

# confusion matrix
print('CONFUSION MATRIX')
h2o.confusionMatrix(rf.model_2)

# predictions
preds_2 = as.data.frame(h2o.predict(object = rf.model_2, newdata = test.hex))
preds_raw_2 = preds_2$predict
preds_raw_2_2 = ifelse(preds_2$p1 > 0.5, 1, 0)

# accuracy
print('ACCURACY (second one is with p = 0.5')
sum(preds_raw_2==y3_test_dum)/length(y3_test_dum)
sum(preds_raw_2_2==y3_test_dum)/length(y3_test_dum)

#auc
print('AUC')
print(auc(y3_test_dum,preds_2$p1))

# diagnostic plot
png(file='model_diagnostics/m_2_roc1.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
plot(roc(y3_test_dum,preds_2$p1), xlim=c(1,0))
dev.off()

png(file='model_diagnostics/m_2_pred_dist.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
plot_pred_type_distribution(data.frame(pred=preds_2$p1,
                                       actual=y3_test_dum), 0.5)
dev.off()


png(file='model_diagnostics/m_2_roc2.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
roc <- calculate_roc(data.frame(pred=preds_2$p1,
                                actual=y3_test_dum), 1, 1, n = 100)
plot_roc(roc, 0.4, 1, 1)
dev.off()

png(file='model_diagnostics/m_2_roc2_falseneg.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
roc <- calculate_roc(data.frame(pred=preds_2$p1,
                                actual=y3_test_dum), 1, 2, n = 100)
plot_roc(roc, 0.4, 1, 2)
dev.off()

#PRECISION-RECALL
png(file='model_diagnostics/m_2_prec-recall.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
pred2 = prediction(preds_2$p1,y3_test_dum)
RP.perf <- performance(pred2, "prec", "rec")
plot (RP.perf)
dev.off()

png(file='model_diagnostics/m_2_roc3.png', width=5,height=5, units='in',
    type = 'cairo', res=200)
ROC.perf <- performance(pred2, "tpr", "fpr")
plot (ROC.perf)
lines(seq(0,1,0.001), seq(0,1,0.001), lty=2)
dev.off()

# F1
print('PRECISION')
precision <- posPredValue(as.factor(preds_raw_2_2), as.factor(y3_test_dum))
precision

print('RECALL')
recall <- sensitivity(as.factor(preds_raw_2_2), as.factor(y3_test_dum))
recall

print('F1')
F1 <- (2 * precision * recall) / (precision + recall)
F1

sink()
# save model
h2o.saveModel(rf.model_2, path = "h2o_models/m_2", force = FALSE)

# save predictions
predictions_2 = cbind.data.frame(
  all_data_store[all_data_store$date >=
                   as.Date('2014-01-01'),
                 c('restaurant_id', 'date',
                   'latitude', 'longitude', 'vio3')],
  preds_2$p1)
names(predictions_2)[length(names(predictions_2))] = 'pred'
predictions_2$pred_raw = ifelse(predictions_2$pred > 0.5, 1, 0)
predictions_2$vio_raw = ifelse(predictions_2$vio3 >= 1, 1, 0)

sum(predictions_2$pred_raw ==
      predictions_2$vio_raw)/nrow(predictions_2)

write.csv(predictions_2,'finalData/predictions_2.csv',
          row.names = F)

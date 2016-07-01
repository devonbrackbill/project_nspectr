# MODEL 4 BOOTSTRAP VALIDATION SET

setwd('C:/Users/Devon/Dropbox/insight/project')
predictions_5 = read.csv('finalData/predictions_final.csv')
predictions_5$restaurant_id = as.character(predictions_5$restaurant_id)
predictions_5$date = as.Date(as.character(predictions_5$date))


sum(predictions_5$pred_raw == predictions_5$vio_raw)/nrow(predictions_5)


N = nrow(predictions_5)
K=10000

set.seed(1234)
samp_mat = sample(1:nrow(predictions_5), K*N, replace=T)

pred_mat = matrix(predictions_5$pred_raw[samp_mat], ncol=K)
actual_mat = matrix(predictions_5$vio_raw[samp_mat], ncol=K)

out = rep(0,K)
for (i in 1:ncol(pred_mat)){
  out[i] = sum(pred_mat[,i] == actual_mat[,i])/N
}

head(out)

quantile(out,c(0.025, 0.975))

# helper functions
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
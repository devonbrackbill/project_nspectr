# CREATES FINAL MAP DATA WITH ACTUAL PREDICTIONS FOR RESTAURANTS

predictions_5 = read.csv('finalData/predictions_final.csv')
predictions_5$restaurant_id = as.character(predictions_5$restaurant_id)

# to add in restaurant names
df = data.frame(read_csv('finalData/train_labels_no_null_with_business_info.csv'))
df$date = as.Date(df$date)

df2 = data.frame(df %>%
                   filter(date >= as.Date('2014-01-01')) %>% 
                   group_by(restaurant_id) %>%
                   slice(which.max(date)))

df2 = df2[,c('name', 'restaurant_id')]

df3 = merge(predictions_5, df2, all.x=T, all.y=F,
            by = c('restaurant_id'))
head(df3)

df4 = data.frame(df3  %>% 
                   group_by(restaurant_id) %>% 
                   slice(which.max(date)))

# accuracy on 1232 unique restaurants
with(df4, sum(vio_raw == pred_raw))/nrow(df4)

write.csv(df4, file='finalData/dataForMapsFinal.csv')

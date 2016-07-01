# PREP DATA
#############

# this is the main data prep module
# it takes the processed data from the python tf-idf matrix
# and conducts all joins, and prepares for all analysis.

rm(list=ls())

# LIBRARIES
library(readr)
library(dplyr)

##############################################
# LOAD DATA
##############################################
setwd('../../')

# Yelp -> tf-idf data (2-grams)
tf_idf = read_csv('finalData/train_mat_time_CORRECTED2.csv')

# Yelp business descriptions
train_labels = read_csv('finalData/train_labels_no_null_with_business_info.csv')

dim(train_labels)
dim(tf_idf)

# rename variables in tf_idf
names(tf_idf) = paste0(names(tf_idf), '_text')

# cbind data, because it's in the correct order
all_data = cbind(tf_idf,train_labels)
dim(all_data)

##############################################
# CLEAN DATA
##############################################

# rename *, **, and *** to proper names
names(all_data)[5005:5007] = c('vio1', 'vio2', 'vio3')
names(all_data)[5009:5011] = c('vio1_lag', 'vio2_lag', 'vio3_lag')


##############################################
# STATISTICS
##############################################

# number with an average >1 violation for each level
out = data.frame(
  all_data %>% 
    group_by(restaurant_id) %>% 
    summarise(perc_vio1 = mean(vio1) > 1 ,
              perc_vio2 = mean(vio2) > 1 ,
              perc_vio3 = mean(vio3) > 1 ) )

sum(out$perc_vio1)/nrow(out)
sum(out$perc_vio2)/nrow(out)
sum(out$perc_vio3)/nrow(out)

##############################################
### FEATURE ENGINEER ###
##############################################

# month and year dummies
all_data$year = as.numeric(format(all_data$date,"%Y"))
all_data$month = as.numeric(format(all_data$date, '%m'))
  
sink('temp.txt')
unlist(lapply(all_data,class))
sink()

# find all non-numeric
non_numeric = names(all_data[!sapply(all_data, is.numeric)])

lapply(all_data[non_numeric], class)

#drop needless columns
all_data = all_data[,-which(names(all_data) %in%
                      c('city', 'full_address', 'hours', 'name', 'state',
                        'type'))]

# re-run non-numeric
non_numeric = names(all_data[!sapply(all_data, is.numeric)])

# special procedure for each remaining variable

#all_data['date']
#all_data['restaurant_id']
#all_data['prior_date']
head(all_data$open) = ifelse(all_data$open == 'True', 1, 0)
all_data$open = as.numeric(all_data$open)

convertToCategorical = function(x){
  
  x[is.na(x)==T] = -1
  
  unique_levels = unlist(unique(x))
  #print(unique_levels)
  counter = 0
  for (lev in unique_levels){

    if (lev == -1){ next  }
    print(paste(lev, '-->',counter))
    x[x == lev] = counter
    counter = counter + 1
  }
  
  #x = as.character(x)
  return(as.numeric(x[,1]))
}


all_data['Accepts Credit Cards'] = convertToCategorical(all_data['Accepts Credit Cards'])
all_data['Ages Allowed'] = convertToCategorical(all_data['Ages Allowed'])
all_data['Alcohol'] = convertToCategorical(all_data['Alcohol'])
all_data['Attire'] = convertToCategorical(all_data['Attire'])
all_data['BYOB'] = convertToCategorical(all_data['BYOB'])
all_data['BYOB/Corkage'] = convertToCategorical(all_data['BYOB/Corkage'])
all_data['By Appointment Only'] = convertToCategorical(all_data['By Appointment Only'])
all_data['Caters'] = convertToCategorical(all_data['Caters'])
all_data['Coat Check'] = convertToCategorical(all_data['Coat Check'])
all_data['Corkage'] = convertToCategorical(all_data['Corkage'])
all_data['Delivery'] = convertToCategorical(all_data['Delivery'])
all_data['Dietary Restrictions'] = convertToCategorical(all_data['Dietary Restrictions'])
all_data['Dogs Allowed'] = convertToCategorical(all_data['Dogs Allowed'])
all_data['Drive-Thru'] = convertToCategorical(all_data['Drive-Thru'])
all_data['Good For Dancing'] = convertToCategorical(all_data['Good For Dancing'])
all_data['Good For Groups'] = convertToCategorical(all_data['Good For Groups'])
all_data['Good For Kids'] = convertToCategorical(all_data['Good For Kids'])
all_data['Good for Kids'] = convertToCategorical(all_data['Good for Kids'])
all_data['Happy Hour'] = convertToCategorical(all_data['Happy Hour'])
all_data['Has TV'] = convertToCategorical(all_data['Has TV'])
all_data['Music'] = convertToCategorical(all_data['Music'])
all_data['Noise Level'] = convertToCategorical(all_data['Noise Level'])
all_data['Open 24 Hours'] = convertToCategorical(all_data['Open 24 Hours'])
all_data['Order at Counter'] = convertToCategorical(all_data['Order at Counter'])
all_data['Outdoor Seating'] = convertToCategorical(all_data['Outdoor Seating'])
all_data['Payment Types'] = convertToCategorical(all_data['Payment Types'])
all_data['Smoking'] = convertToCategorical(all_data['Smoking'])
all_data['Take-out'] = convertToCategorical(all_data['Take-out'])
all_data['Takes Reservations'] = convertToCategorical(all_data['Takes Reservations'])
all_data['Waiter Service'] = convertToCategorical(all_data['Waiter Service'])
all_data['Wheelchair Accessible'] = convertToCategorical(all_data['Wheelchair Accessible'])
all_data['Wi-Fi'] = convertToCategorical(all_data['Wi-Fi'])
all_data['casual'] = convertToCategorical(all_data['casual'])
all_data['classy'] = convertToCategorical(all_data['classy'])
all_data['divey'] = convertToCategorical(all_data['divey'])
all_data['hipster'] = convertToCategorical(all_data['hipster'])
all_data['intimate'] = convertToCategorical(all_data['intimate'])
all_data['romantic'] = convertToCategorical(all_data['romantic'])
all_data['touristy'] = convertToCategorical(all_data['touristy'])
all_data['trendy'] = convertToCategorical(all_data['trendy'])
all_data['upscale'] = convertToCategorical(all_data['upscale'])
all_data['garage'] = convertToCategorical(all_data['garage'])
all_data['lot'] = convertToCategorical(all_data['lot'])
all_data['street'] = convertToCategorical(all_data['street'])
all_data['valet'] = convertToCategorical(all_data['valet'])
all_data['validated'] = convertToCategorical(all_data['validated'])
all_data['breakfast'] = convertToCategorical(all_data['breakfast'])
all_data['brunch'] = convertToCategorical(all_data['brunch'])
all_data['dessert'] = convertToCategorical(all_data['dessert'])
all_data['dinner'] = convertToCategorical(all_data['dinner'])
all_data['latenight'] = convertToCategorical(all_data['latenight'])
all_data['lunch'] = convertToCategorical(all_data['lunch'])

# re-run non-numeric
non_numeric = names(all_data[!sapply(all_data, is.numeric)])
non_numeric

#drop
all_data = all_data[,-which(names(all_data) %in% c('prior_date'))]

# dummy for each restaurant_id
# just use train_X to train and test_X to validate
temp = as.factor(all_data$restaurant_id)
temp = data.frame(temp=temp)
dum = dummyVars(~temp, data=temp)
trsf <- data.frame(predict(dum, newdata = temp))

all_data = cbind(all_data, trsf)

write.csv(all_data,file='finalData/DataPriorToModels.csv')

sink('temp.txt')
unlist(lapply(all_data,class))
sink()

sink('names.txt')
names(all_data)
sink()

all_data_cut = all_data[,-c(1:5002,5278:7128)]

write.csv(all_data_cut, file='finalData/DataPriorToModels_cut.csv')

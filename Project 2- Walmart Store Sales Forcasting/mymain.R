rm(list = ls())
ls()
search()
library(magrittr)
library(tidyverse)
library(dplyr)
# install.packages('lubridate')
library(lubridate)


preprocess.svd <- function(train, n.comp){
  # train$Date <- as.Date(train$Date,format="%m/%d/%Y")
  train <- train %>%
    select(Store, Dept, Date, Weekly_Sales) %>%
    spread(Date, Weekly_Sales)
  train[is.na(train)] <- 0

  train_svd = NULL

  for(mydept in unique(train$Dept)){
    dept_data <- train %>%
      filter(Dept == mydept)

    if (nrow(dept_data) > n.comp){
      tmp_data <- dept_data[, -c(1,2)]
      store_means <- rowMeans(tmp_data)
      tmp_data <- tmp_data - store_means
      z <- svd(tmp_data, nu=n.comp, nv=n.comp)
      s <- diag(z$d[1:n.comp])
      tmp_data <- z$u %*% s %*% t(z$v) + store_means
      tmp_data[tmp_data < 0] <- 0
      dept_data[, -c(1:2)] <- z$u %*% s %*% t(z$v) + store_means
    }
    train_svd = rbind(train_svd, dept_data)
  }
  train_svd <- train_svd %>%
    gather(Date, Weekly_Sales, -Store, -Dept)
  return(train_svd)
}

mypredict = function(){
  # train$Date <- ymd(train$Date)
  # train$Date <- mdy(train$Date)
  # train$Date <- as.Date(train$Date,format="%m/%d/%Y")
  
  train_svd = preprocess.svd(train, 8)
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>%
    filter(Date >= start_date & Date < end_date) %>%
    select(-IsHoliday)

  start_last_year = min(test_current$Date) - 375
  end_last_year = max(test_current$Date) - 350
  tmp_train <- train_svd %>%
    filter(Date > start_last_year & Date < end_last_year) %>%
    mutate(Wk = ifelse(year(Date) == 2010, week(Date)-1, week(Date))) %>%
    rename(Weekly_Pred = Weekly_Sales) %>%
    select(-Date)

  test_current <- test_current %>%
    mutate(Wk = week(Date))

    # not all depts need prediction
    test_depts <- unique(test_current$Dept)
    test_pred <- NULL

    for(dept in test_depts){
      train_dept_data <- train_svd %>% filter(Dept == dept)
      test_dept_data <- test_current %>% filter(Dept == dept)

      # no need to consider stores that do not need prediction
      # or do not have training samples
      train_stores <- unique(train_dept_data$Store)
      test_stores <- unique(test_dept_data$Store)
      test_stores <- intersect(train_stores, test_stores)

      for(store in test_stores){
        tmp_train <- train_dept_data %>%
          filter(Store == store) %>%
          mutate(Wk = ifelse(year(Date) == 2010, week(Date)-1, week(Date))) %>%
          mutate(Yr = year(Date))
        tmp_test <- test_dept_data %>%
          filter(Store == store) %>%
          mutate(Wk = ifelse(year(Date) == 2010, week(Date)-1, week(Date))) %>%
          mutate(Yr = year(Date))

        tmp_train$Wk = factor(tmp_train$Wk, levels = 1:52)
        tmp_test$Wk = factor(tmp_test$Wk, levels = 1:52)

        train_model_matrix <- model.matrix(~ Yr + Wk, tmp_train)
        test_model_matrix <- model.matrix(~ Yr + Wk, tmp_test)
        mycoef <- lm(tmp_train$Weekly_Sales ~ train_model_matrix)$coef
        mycoef[is.na(mycoef)] <- 0
        tmp_pred <- mycoef[1] + test_model_matrix %*% mycoef[-1]

        tmp_test <- tmp_test %>%
          mutate(Weekly_Pred = tmp_pred[,1]) %>%
          select(-Wk, -Yr)
        test_pred <- test_pred %>% bind_rows(tmp_test)
      }
    }
    return(test_pred)
}



##### Evaluation #####
# read in train / test dataframes
train <- readr::read_csv('train_ini.csv')
# train$Date <- mdy(train$Date)
train$Date <- as.Date(train$Date,format="%m/%d/%Y")
test <- readr::read_csv('test.csv')

# save weighted mean absolute error WMAE
num_folds <- 10
wae <- rep(0, num_folds)

for (t in 1:num_folds) {
  # *** THIS IS YOUR PREDICTION FUNCTION ***
  test_pred <- mypredict()

  # load fold file
  fold_file <- paste0('fold_', t, '.csv')
  new_train <- readr::read_csv(fold_file,
                               col_types = cols())
  train <- train %>% add_row(new_train)

  # extract predictions matching up to the current fold
  scoring_tbl <- new_train %>%
    left_join(test_pred, by = c('Date', 'Store', 'Dept'))

  # compute WMAE
  actuals <- scoring_tbl$Weekly_Sales
  preds <- scoring_tbl$Weekly_Pred
  preds[is.na(preds)] <- 0
  weights <- if_else(scoring_tbl$IsHoliday, 5, 1)
  wae[t] <- sum(weights * abs(actuals - preds)) / sum(weights)
}

print(wae)
mean(wae)










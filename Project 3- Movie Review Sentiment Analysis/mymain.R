#####################################
# load libraries
library(rsparse)
library(Rcpp)
library(text2vec)
library(glmnet)
library(pROC)
####################################



#####################################
# Load your vocabulary and training data
#####################################
myvocab <- scan(file = "myvocab.txt", what = character())
train <- read.table("train.tsv", stringsAsFactors = FALSE,
                    header = TRUE)

#####################################
# Train a binary classification model

train = read.table("train.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)
train$review <- gsub('<.*?>', ' ', train$review)
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)


fit1 = cv.glmnet(x = dtm_train, 
                 y = train$sentiment, 
                 alpha = 0,
                 family='binomial')

#####################################


test <- read.table("test.tsv", stringsAsFactors = FALSE,
                   header = TRUE)

#####################################
# Compute prediction 
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predited probabilities


test$review <- gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 2L)))
dtm_test = create_dtm(it_test, vectorizer)


output = predict(fit1, dtm_test, s=fit1$lambda.min, type = 'response')
test$prediction = output
output = data.frame(test$id, test$prediction)
names(output) = c('id', 'prob')

#####################################

write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')
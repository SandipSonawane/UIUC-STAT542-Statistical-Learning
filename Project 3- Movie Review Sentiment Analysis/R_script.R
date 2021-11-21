# load training data, clean html tags

j = 1
setwd(paste("split_", j, sep=""))
train = read.table("train.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)
train$review = gsub('<.*?>', ' ', train$review)

# Next, use R package text2vec to construct DT (DocumentTerm) matrix (maximum 4-grams).
# The default vocabulary size (i.e., # of columns of dtm_train) is more than 30,000, 
# bigger than the sample size n = 25,000.
# install.packages('rsparse')
install.packages(c('float', 'rsparse'), type = 'source')
library(rsparse)
library(Rcpp)
library(text2vec)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "the", "us")
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
tmp.vocab = create_vocabulary(it_train, 
                              stopwords = stop_words, 
                              ngram = c(1L,4L))
tmp.vocab = prune_vocabulary(tmp.vocab, term_count_min = 10,
                             doc_proportion_max = 0.5,
                             doc_proportion_min = 0.001)
dtm_train  = create_dtm(it_train, vocab_vectorizer(tmp.vocab))                                                                                                                 
                                                                                                                   stop_words = c("i", "me", "my", "myself", 
                                                                                                                                  "we", "our", "ours", "ourselves", 
                                                                                                                                  "you", "your", "yours", 
                                                                                                                                  "their", "they", "his", "her", 
                                                                                                                                  "she", "he", "a", "an", "and",
                                                                                                                                  "is", "was", "are", "were", 
                                                                                                                                  "him", "himself", "has", "have", 
                                                                                                                                  "it", "its", "the", "us")
                                                                                                                   it_train = itoken(train$review,
                                                                                                                                     preprocessor = tolower, 
                                                                                                                                     tokenizer = word_tokenizer)
                                                                                                                   tmp.vocab = create_vocabulary(it_train, 
                                                                                                                                                 stopwords = stop_words, 
                                                                                                                                                 ngram = c(1L,4L))
                                                                                                                   tmp.vocab = prune_vocabulary(tmp.vocab, term_count_min = 10,
                                                                                                                                                doc_proportion_max = 0.5,
                                                                                                                                                doc_proportion_min = 0.001)
                                                                                                                   dtm_train  = create_dtm(it_train, vocab_vectorizer(tmp.vocab))
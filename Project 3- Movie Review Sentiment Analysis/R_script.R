# load training data, clean html tags

j = 1
setwd(paste("split_", j, sep=""))
train = read.table("train.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)
train$review = gsub('<.*?>', ' ', train$review)

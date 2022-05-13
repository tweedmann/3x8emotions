
##################################################################
##################################################################
## Widmann & Wich: Creating and Comparing Dictionary, Word Embedding, and Transformer-based 
## Models to Measure Discrete Emotions in German Political Text
## Political Analysis
## widmann@ps.au.dk
##################################################################
##################################################################

#### Applying Neural Network Classifiers based on Locally Trained German Word Embeddings

# Load necessary packages
library(quanteda)
library(corpus)
library(keras)
library(tidytext)

# Set working directory
setwd("./neuralnet")


# First, you need to turn your text into sentences
data <- data %>% 
  unnest_tokens(sentences, text, "sentences")

# Now, you can turn your text documents in a corpus
corp <- corpus(data$sentences)

# Create a document feature matrix and conduct pre-processing
text_dfm <- dfm(corp, remove=stopwords("german"), verbose=TRUE, tolower = TRUE)

# Stemming
text_dfm <- dfm_wordstem(text_dfm, language = "german")

# Now, we will convert the word embeddings into a data frame 
# and match the features from each document with their corresponding embeddings

#F irst, we load the locally trained word embeddings into R
w2v <- readr::read_delim("./vec_ed_preprocessed.txt", 
                         skip=1, delim=" ", quote="",
                         col_names=c("word", paste0("V", 1:100)))

# Stem the terms included in the embeddings to increase matches
w2v$word <- text_tokens(w2v$word, stemmer = "de")

# creating new feature matrix for embeddings
embed <- matrix(NA, nrow=ndoc(text_dfm), ncol=100)
for (i in 1:ndoc(cgdfm)){
  if (i %% 100 == 0) message(i, '/', ndoc(text_dfm))
  # extract word counts
  vec <- as.numeric(text_dfm[i,])
  # keep words with counts of 1 or more
  doc_words <- featnames(text_dfm)[vec>0]
  # extract embeddings for those words
  embed_vec <- w2v[w2v$word %in% doc_words, 2:101]
  # aggregate from word- to document-level embeddings by taking AVG
  embed[i,] <- colMeans(embed_vec, na.rm=TRUE)
  # if no words in embeddings, simply set to 0
  if (nrow(embed_vec)==0) embed[i,] <- 0
}

# After you created the sentence embeddings, you can apply the trained machine learning models for each emotion
# The machine learning models are provided in the folder "./neuralnet/models"
# for example, anger:

model <- load_model_hdf5("./models/keras_anger", custom_objects = NULL, compile = TRUE)
wb.anger <- model %>% predict_classes(embed)
data <- cbind(data, wb.anger)
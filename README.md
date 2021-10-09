3x8emotions
================
Tobias Widmann & Maximilian Wich
October 15, 2021

Repo containing code and models for 3 different tools to measure appeals
to 8 discrete emotions in German political text, as described and
validated in the following article:

Please start by reading this article which contains information about
the creation and performance of the different tools. These tools are
free to use for academic research. In case you use one or multiple of
these, please always cite the article above.

In order to obtain all necessary files, start by downloading this repo
as a .zip file. The folder contains all scripts to apply the (1) ed8
dictionary, (2) the neural network models based on locally trained word
embeddings and (3) the ELECTRA model.

## (1) ed8

The `ed8 dictionary` is provided in YAML format and can be applied via
the the `quanteda` package. The dictionary and the R script to apply the
dictionary to a data frame with a ‘text’ column can be found in the
folder “./ed8”.

``` r
# First, load quanteda package
library(quanteda)

#  Load in dictionary
ed8 <- dictionary(file = "./ed8.yml",
                  format = "YAML")

# Create the function
get_ed8_emotions <- function(data){
  #Create a corpus from your data frame
  corp <- corpus(data)
  
  #Tokenize corpus and pre-process (remove punctuations, numbers, and urls)
  toks <- tokens(corp, remove_punct = TRUE, remove_numbers = TRUE, remove_url = TRUE)
  
  #Create DFM just to measure number of terms before removing stopwords
  terms_dfm <- dfm(toks)
  
  #Create bigram-compounds to include negation control
  toks_neg_bigram <- tokens_compound(toks, pattern = phrase("nicht *"))
  toks_neg_bigram <- tokens_compound(toks_neg_bigram, pattern = phrase("nichts *"))
  toks_neg_bigram <- tokens_compound(toks_neg_bigram, pattern = phrase("kein *"))
  toks_neg_bigram <- tokens_compound(toks_neg_bigram, pattern = phrase("keine *"))
  toks_neg_bigram <- tokens_compound(toks_neg_bigram, pattern = phrase("keinen *"))
  
  #Turn tokens into DFM, remove stopwords
  emo_dfm <- dfm(toks_neg_bigram, remove = stopwords("de"))
  
  #Apply dictionary
  dict_dfm_results <- dfm_lookup(emo_dfm,ed8)
  
  #Convert results back to data frame
  results_df <- cbind(data, convert(dict_dfm_results, to = 'data.frame'))
  
  #Assign length to each documents
  results_df$terms_raw <- ntoken(terms_dfm)
  results_df$terms <- ntoken(emo_dfm)
  
  return(results_df)
}

# Now you can use the function on your data; simply enter a data frame with a column called "text" including the text data
results <- get_ed8_emotions(data)

# Finally, you can create normalized emotional scores by dividing the ed8-scores by document length
results$anger.norm <- results$ed8.ANGER / results$terms
results$fear.norm <- results$ed8.FEAR / results$terms
results$disgust.norm <- results$ed8.DISGUST / results$terms
results$sadness.norm <- results$ed8.SADNESS / results$terms
results$joy.norm <- results$ed8.JOY / results$terms
results$enthusiasm.norm <- results$ed8.ENTHUSIASM / results$terms
results$pride.norm <- results$ed8.PRIDE / results$terms
results$hope.norm <- results$ed8.HOPE / results$terms
```

## (2) Neural Network Classifiers

The neural network classifiers and locally trained word embedding model
are provided in the folder “./neuralnet”. The code for turning text into
numerical vectors and subsequently applying the neural network
classifiers can be found in the R script `apply_neuralnet.Rscript`.

``` r
# Load necessary packages
library(quanteda)
library(corpus)
library(keras)

# Set working directory
setwd("./neuralnet")

# First, you need to turn your text documents in a corpus
cgcorpus <- corpus(data$text)

# Create a document feature matrix and conduct pre-processing
cgdfm <- dfm(cgcorpus, remove=stopwords("german"), verbose=TRUE, tolower = TRUE)

# Stemming
cgdfm <- dfm_wordstem(cgdfm, language = "german")

# Now, we will convert the word embeddings into a data frame 
# and match the features from each document with their corresponding embeddings

#F irst, we load the locally trained word embeddings into R
w2v <- readr::read_delim("./vec_ed_preprocessed.txt", 
                         skip=1, delim=" ", quote="",
                         col_names=c("word", paste0("V", 1:100)))

# Stem the terms included in the embeddings to increase matches
w2v$word <- text_tokens(w2v$word, stemmer = "de")

# Match embeddings with DFM
w2v <- w2v[w2v$word %in% featnames(cgdfm),]

# creating new feature matrix for embeddings
embed <- matrix(NA, nrow=ndoc(cgdfm), ncol=100)
for (i in 1:ndoc(cgdfm)){
  if (i %% 100 == 0) message(i, '/', ndoc(cgdfm))
  # extract word counts
  vec <- as.numeric(cgdfm[i,])
  # keep words with counts of 1 or more
  doc_words <- featnames(cgdfm)[vec>0]
  # extract embeddings for those words
  embed_vec <- w2v[w2v$word %in% doc_words, 2:101]
  # aggregate from word- to document-level embeddings by taking AVG
  embed[i,] <- colMeans(embed_vec, na.rm=TRUE)
  # if no words in embeddings, simply set to 0
  if (nrow(embed_vec)==0) embed[i,] <- 0
}

# After we created the sentence embeddings, we apply the trained machine learning models for each emotion
# The machine learning models are provided in the folder "./neuralnet/models"

model <- load_model_hdf5("./models/keras_anger", custom_objects = NULL, compile = TRUE)
wb.anger <- model %>% predict_classes(embed)
data <- cbind(data, wb.anger)
```

## (3) ELECTRA Model

The ELECTRA files are provided in the folder “./electra”. The model can
be applied to text data using the Python code as shown in
`apply_electra.ipynb`. Simply use a data frame with text column.

``` python
# Set working directory
%cd /electra/
```

``` python
# load necessary modules
import transformers
import torch
import sklearn
import pandas as pd
```

``` python
# text documents the model will be applied to
documents = data.text
```

``` python
# load inferencer
from helper.inferencing import Inferencer
```

``` python
# predicting
predictor = Inferencer()
df_results = predictor.predict_dataframe(documents)
```

``` python
# show results
df_results
```

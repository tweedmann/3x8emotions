##################################################################
##################################################################
## Widmann & Wich: Creating and Comparing Dictionary, Word Embedding, and Transformer-based 
## Models to Measure Discrete Emotions in German Political Text
## Political Analysis
## widmann@ps.au.dk
##################################################################
##################################################################


#### Applying ed8 dictionary

#### FUNCTION ed8 #################################

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

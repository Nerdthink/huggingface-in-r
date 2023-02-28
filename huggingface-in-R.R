#install and load the required packages 
install.packages(c("text","reticulate","dplyr","keras","tm","tidyr", "wordcloud", "ggplot"))

#Load the required packages
library(text)
library(reticulate)
library(dplyr)
library(tm)
library(tidyr)
library(wordcloud)
library(ggplot2)

####Install miniconda to create a python environment in rstuido
install_miniconda(path = miniconda_path(), update = TRUE, force = FALSE)

##Using conda to create python environment
reticulate::conda_create()

## installing keras in your python env
install_keras( 
  method = c("auto", "virtualenv", "conda"), 
  conda = "auto", 
  version = "default", 
  extra_packages = NULL, 
  pip_ignore_installed = TRUE )

library(keras)
###Use reticulate to install transformers in R python environment
reticulate::py_install("transformers")

### import transformers and touch 
# Load the required Python libraries
transformers <- import("transformers")
torch <- import("torch")

#load the pretrained model
model_name <- "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer <- transformers$BertTokenizer$from_pretrained(model_name)
model <- transformers$BertForSequenceClassification$from_pretrained(model_name)

##Load the Restaurant review dataset in tsv.
# Load and preprocess the restaurant review dataset
dataset <- read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

##Preprocess the data
reviews <- tolower(reviews)
reviews <- removePunctuation(reviews)
labels <- dataset$Liked


# Input processing
inputs <- lapply(reviews, tokenizer$encode_plus, add_special_tokens = TRUE, return_tensors = "pt")

# Model inference
outputs <- lapply(inputs, function(input) {
  with(torch$no_grad(), {
    model(input_ids = input$input_ids, token_type_ids = input$token_type_ids, attention_mask = input$attention_mask)$logits
  })
})

# Prediction
predictions <- sapply(outputs, function(output) {
  torch$argmax(output)$item()
})



# Convert sentiment scores to binary scores
threshold <- 2.5
dataset$binary_scores <- ifelse(predictions >= threshold, 1, 0)

# Show the sentiment analysis results
dataset %>% 
  mutate(sentiment = ifelse(dataset$binary_scores > 0, "Positive", "Negative")) %>%
  group_by(sentiment) %>%
  summarise(count = n())

dataset$sentimentresult <- ifelse(dataset$binary_scores >0 , "Positive", "Negative")
dataset$sentimentresult


# Evaluate the sentiment analysis results
confusion_matrix <- table(unlist(labels), unlist(dataset$binary_scores))
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- diag(confusion_matrix) / colSums(confusion_matrix)
recall <- diag(confusion_matrix) / rowSums(confusion_matrix)

confusion_matrix
accuracy
precision
recall



### Creating visuals for the results
library(ggplot2)
# Plot the sentiment analysis results
# Create a bar chart of sentiment counts
# Define colors for positive and negative sentiments
pos_color <- "#1b9e77"
neg_color <- "#d95f02"
    
# Create a bar chart of sentiment counts with colors
ggplot(data = dataset, aes(x = sentimentresult, fill = sentimentresult)) +
  geom_bar() +
  scale_fill_manual(values = c(pos_color, neg_color)) +
  labs(title = "Sentiment Analysis Results", x = "Sentiment", y = "Count")
  
# Split reviews into positive and negative categories
pos_reviews <- dataset %>% filter(binary_scores == 1) %>% pull(Review)
neg_reviews <- dataset %>% filter(binary_scores == 0) %>% pull(Review)

# Create wordclouds of frequent words in positive and negative reviews
# positive reviews
pos_words <- pos_reviews %>% 
  paste(collapse = " ") %>%
  removeWords(stopwords(kind = "en")) %>%
  strsplit(split = "\\s+") %>%
  unlist() %>%
  table() %>%
  as.data.frame() %>%
  setNames(c("word", "freq"))

neg_words <- neg_reviews %>% 
  paste(collapse = " ") %>%
  removeWords(stopwords(kind = "en")) %>%
  strsplit(split = "\\s+") %>%
  unlist() %>%
  table() %>%
  as.data.frame() %>%
  setNames(c("word", "freq"))


wordcloud(words = pos_words$word, freq = pos_words$freq, scale=c(4, 0.5), 
          min.freq = 2, max.words=100, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

wordcloud(words = neg_words$word, freq = neg_words$freq, scale=c(4, 0.5), 
          min.freq = 2, max.words=100, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

 



### Using the text() package
##Install text required python packages (rpp) in a self-contained environment. 
textrpp_install()

##Initialize text required python packages to call from R.
textrpp_initialize()

classification <- textClassify(dataset,model = "distilbert-base-uncased-finetuned-sst-2-english", set_seed = 1234, return_incorrect_results = TRUE,
                               ,function_to_apply = "softmax")
classification
comment(classification)

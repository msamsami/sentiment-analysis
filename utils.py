from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
import re, string
import nltk
import streamlit as st

st.cache(show_spinner=False)
def initialize_():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# This function removes useless information from the input tokens
def remove_noise(tweet_tokens, stop_words = ()):    
    initialize_()
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        # Remove hyperlinks
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        # Remove "@" mentions
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        # Determine position for lemmatizing
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        # Word lemmatizing
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

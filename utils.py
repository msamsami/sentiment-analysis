import re
import string
from typing import Optional, Sequence

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag


def remove_noise(tokens, stop_words: Optional[Sequence] = None):
    """
    Cleans the input tokens by removing useless information such as hyperlinks, mentions, and stopwords.
    It also performs word lemmatizing.
    """
    stop_words = stop_words or []
    
    cleaned_tokens = []

    for token, tag in pos_tag(tokens):
        # Remove hyperlinks
        token = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"
            "(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            token,
        )

        # Remove "@" mentions
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        # Determine position for lemmatizing
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"

        # Word lemmatizing
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if (
            len(token) > 0
            and token not in string.punctuation
            and token.lower() not in stop_words
        ):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

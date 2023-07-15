import pickle

import nltk
from nltk.tokenize import word_tokenize
import streamlit as st

from utils import remove_noise


def get_text():
    """
    Gets user input text/comment.
    """
    input_text = st.text_input(
        "Your comment: ", "I really like this product! It's awesome."
    )
    return input_text


@st.cache(show_spinner=False)
def _initialize_model():
    """
    Initializes the sentiment classification model.
    """
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")

    with open("models/naive_bayes.mdl", "rb") as file:
        model = pickle.load(file)

    return model


def main():
    # Set page config
    st.set_page_config(
        page_title="Sentiment Analysis",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="auto",
    )

    # Set page title text
    st.title(
        """
    Comment Sentiment Analysis  
    This app will detect the sentiment of an user's comment as either positive or negative.
    """
    )

    # Initialize sidebar
    st.sidebar.title("Details")
    st.sidebar.text("")

    # Sidebar information
    st.sidebar.text("Preprocessing:")
    st.sidebar.markdown(
        """
                    * URL removal
                    * @ Mention removal
                    * Lemmatization
                    * Tokenization
                    """
    )
    st.sidebar.text("")
    st.sidebar.text("Model: ")
    st.sidebar.text("Naive Bayes Classifier")

    input_comment = get_text()  # Get user input

    classifier = _initialize_model()

    if (not input_comment) or (input_comment.isspace()):
        st.write("Write a comment and press the Enter key...")
    else:
        input_tokens = remove_noise(word_tokenize(input_comment.replace("'", "")))
        dist = classifier.prob_classify(dict([token, True] for token in input_tokens))
        prob = [dist.prob(label) for label in dist.samples()]

        confidence = max(prob)
        if prob[0] > prob[1]:
            st.image("images/positive.png", width=200)
        else:
            st.image("images/negative.png", width=200)
        st.write("Confidence score = ", float(str(confidence)[:6]))


if __name__ == "__main__":
    main()

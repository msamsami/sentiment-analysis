import streamlit as st
from nltk.tokenize import word_tokenize
from preprocessing import remove_noise
import pickle
import nltk

#----------- A simple function to get user input
def get_text():
    input_text = st.text_input("Your comment: ", "I really like this product! It's awesome.")
    return input_text
#-----------------------------------------------

#----------------------- Initialization function
@st.cache(show_spinner=False)
def initialize_():  
    nltk.download('punkt')
    
    # Load model file
    with open('models/naive_bayes.mdl', 'rb') as file:
        classifier = pickle.load(file)

    return classifier
#-----------------------------------------------

def main():
    # Set page config
    st.set_page_config(page_title="Sentiment Analysis", page_icon=None,
                       layout='centered', initial_sidebar_state='auto')
    
    # Set page title text
    st.title("""
    Comment Sentiment Analysis  
    This app will detect the sentiment of an user's comment as either positive or negative.
    """)
    
    # Initialize sidebar
    st.sidebar.title("Details")
    st.sidebar.text("")

    # Sidebar information
    st.sidebar.text("Preprocessing:")
    st.sidebar.markdown("""
                    * URL removal
                    * @ Mention removal
                    * Lemmatization
                    * Tokenization
                    """)
    st.sidebar.text("")
    st.sidebar.text("Model: ")
    st.sidebar.text("Naive Bayes Classifier")
    
    input_comment = get_text()  # Get user input
    
    classifier = initialize_()
    
    if True:
        if (not input_comment) or (input_comment.isspace()):  # If input was empty
            st.write("Write a comment and press the Enter key...")
        else:
            input_tokens = remove_noise(word_tokenize(input_comment.replace("'", "")))
            dist = classifier.prob_classify(dict([token, True] for token in input_tokens))
            prob = [dist.prob(label) for label in dist.samples()]
            
            confidence = max(prob)
            if prob[0]>prob[1]:
                st.image("images/positive.png", width=200)
            else:
                st.image("images/negative.png", width=200)
            st.write("Confidence score = ", float(str(confidence)[:6]))
    else:
        st.write("Something went wrong!")
    #-----------------------------------------------
    
if __name__ == "__main__":
    main()

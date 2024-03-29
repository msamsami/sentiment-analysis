import argparse
import pickle
import random

import nltk
from nltk.corpus import twitter_samples
from nltk import FreqDist, classify, NaiveBayesClassifier

from utils import remove_noise


def get_all_words(list_tokens):
    for tokens in list_tokens:
        for token in tokens:
            yield token


def get_tweets_for_model(list_tokens):
    for tweet_tokens in list_tokens:
        yield dict([token, True] for token in tweet_tokens)


def train(save_to: str, test_size: float = 0.3):
    # Download necessary modules
    nltk.download("twitter_samples")
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download("stopwords")
    nltk.download("punkt")

    # Load tokenized positive and negative tweets
    positive_tweet_tokens = twitter_samples.tokenized("positive_tweets.json")
    negative_tweet_tokens = twitter_samples.tokenized("negative_tweets.json")

    # Initialize two empty lists to store clean tokens (for positive and negative tweets)
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    # # Load English stopwords
    # stop_words = stopwords.words('english')

    # Clean positive tweets' tokens
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, ()))

    # Clean negative tweets' tokens
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, ()))

    # Get all words from the list of clean positive tweets' tokens
    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    # Frequency of all positive words
    freq_dist_pos = FreqDist(all_pos_words)

    # Prepare tokens in dictionary format for model training
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    # Label the positive tokens and construct the positive dataset
    positive_dataset = [
        (tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model
    ]
    # Label the negative tokens and construct the negative dataset
    negative_dataset = [
        (tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model
    ]

    # Train-test split
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)
    train_data = dataset[: round((1 - test_size) * len(dataset))]
    test_data = dataset[round((1 - test_size) * len(dataset)) :]

    # Train the model on data
    classifier = NaiveBayesClassifier.train(train_data)

    # Save the model file
    with open(save_to, "wb") as file:
        pickle.dump(classifier, file)

    # Evaluate test accuracy
    test_accuracy = classify.accuracy(classifier, test_data)
    print("Test accuracy of the trained classifier = {:.4f}".format(test_accuracy * 100))

    # n-most informative features to be returned
    informative_features = classifier.show_most_informative_features(10)
    # print(informative_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--save-to", type=str, required=True)
    args = parser.parse_args()

    train(
        test_size=args.test_size,
        save_to=args.save_to,
    )

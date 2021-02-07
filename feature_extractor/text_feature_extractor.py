import numpy as np
import string

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.data
from nltk.corpus import stopwords


def add_text_feature(df):
    df["split"] = df["text"].apply(nltk.word_tokenize)
    print("split finished...")

    sid = SentimentIntensityAnalyzer()
    df["sid"] = df["text"].apply(sid.polarity_scores)
    for k in ["neu", "compound", "pos", "neg"]:
        df["sid_" + k] = df["sid"].apply(lambda x: x[k])
    print("polarity_scores finished...")

    df["pos_tag"] = df["split"].apply(lambda x: [w[1] for w in nltk.pos_tag(x)])
    for pos in ["CC", "RB", "IN", "NN", "VB", "VBP", "JJ", "PRP", "TO", "DT"]:
        df["n_pos_" + pos] = df["pos_tag"].apply(lambda x: x.count(pos))
    print("pos_tag finished...")

    # Number of words in the text #
    df["num_words"] = df["split"].apply(len)

    # Number of unique words in the text #
    df["num_unique_words"] = df["split"].apply(lambda x: len(set(x)))

    # Number of characters in the text #
    df["num_chars"] = df["text"].apply(len)

    # Number of stopwords in the text #
    df["num_stopwords"] = df["split"].apply(
        lambda x: len([w for w in x if w in set(stopwords.words("english"))])
    )

    # Number of punctuations in the text #
    df["num_punctuations"] = df["split"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )

    # Number of title case words in the text #
    df["num_words_upper"] = df["split"].apply(
        lambda x: len([w for w in x if w.isupper()])
    )

    # Number of title case words in the text #
    df["num_words_title"] = df["split"].apply(
        lambda x: len([w for w in x if w.istitle()])
    )

    # Average length of the words in the text #
    df["mean_word_len"] = df["split"].apply(lambda x: np.mean([len(w) for w in x]))

    anchor_words = [
        "the",
        "a",
        "appear",
        "little",
        "was",
        "one",
        "two",
        "three",
        "ten",
        "is",
        "are",
        "ed",
        "however",
        "to",
        "into",
        "about",
        "th",
        "er",
        "ex",
        "an",
        "ground",
        "any",
        "silence",
        "wall",
    ]

    gender_words = ["man", "woman", "he", "she", "her", "him", "male", "female"]

    for word in anchor_words + gender_words:
        df["n_" + word] = df["split"].apply(
            lambda x: len([w for w in x if w.lower() == word])
        )
    return df

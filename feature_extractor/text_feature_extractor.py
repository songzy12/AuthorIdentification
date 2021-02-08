import numpy as np
import string

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.data
from nltk.corpus import stopwords


def add_text_feature(df):
    sid = SentimentIntensityAnalyzer()
    df["polarity"] = df["text"].apply(sid.polarity_scores)
    for k in ["neu", "compound", "pos", "neg"]:
        df["polarity_" + k] = df["polarity"].apply(lambda x: x[k])
    print("polarity_scores finished...")

    df["words"] = df["text"].apply(nltk.word_tokenize)
    print("word tokenize finished...")

    df["pos_tag"] = df["words"].apply(
        lambda x: [w[1] for w in nltk.pos_tag(x)])
    for pos in ["CC", "RB", "IN", "NN", "VB", "VBP", "JJ", "PRP", "TO", "DT"]:
        df["count_pos_" + pos] = df["pos_tag"].apply(lambda x: x.count(pos))
    print("pos_tag finished...")

    # Number of words in the text #
    df["count_words"] = df["words"].apply(len)

    # Number of unique words in the text #
    df["count_unique_words"] = df["words"].apply(lambda x: len(set(x)))

    # Number of characters in the text #
    df["count_chars"] = df["text"].apply(len)

    # Number of stopwords in the text #
    df["count_stopwords"] = df["words"].apply(
        lambda x: len([w for w in x if w in set(stopwords.words("english"))])
    )

    # Number of punctuations in the text #
    df["count_punctuations"] = df["words"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )

    # Number of title case words in the text #
    df["count_words_upper"] = df["words"].apply(
        lambda x: len([w for w in x if w.isupper()])
    )

    # Number of title case words in the text #
    df["count_words_title"] = df["words"].apply(
        lambda x: len([w for w in x if w.istitle()])
    )

    # Average length of the words in the text #
    df["mean_word_len"] = df["words"].apply(
        lambda x: np.mean([len(w) for w in x]))

    print("count words finished...")

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
    for word in anchor_words:
        df["count_anchor_" + word] = df["words"].apply(
            lambda x: len([w for w in x if w.lower() == word])
        )

    gender_words = ["man", "woman", "he",
                    "she", "her", "him", "male", "female"]
    for word in gender_words:
        df["count_gender_" + word] = df["words"].apply(
            lambda x: len([w for w in x if w.lower() == word])
        )

    print("anchor/gender words finished...")
    return df.drop(columns=['pos_tag', 'polarity', 'words'])

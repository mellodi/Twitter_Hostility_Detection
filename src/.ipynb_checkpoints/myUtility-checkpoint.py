import configparser
import json
import re

import numpy as np
import pandas as pd
import sklearn.feature_extraction as feature_extraction
import tensorflow as tf
from emoji_function import demojize
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics import roc_auc_score
from TwitterAPI import TwitterAPI


def getTwitter(config_file):
    """
        Function to autherize the twitterAPI access
        Params:
            config_file: String representing the config file name,
                including the consumer_key, consumer_secret,
                access_token, and access_token_secret
        Returns:
            twitter: A Twitetr API handle
    """

    config = configparser.ConfigParser()
    config.read(config_file)
    twitter = TwitterAPI(
        config.get("twitter", "consumer_key"),
        config.get("twitter", "consumer_secret"),
        config.get("twitter", "access_token"),
        config.get("twitter", "access_token_secret"),
    )
    return twitter


def filePrepration(rootAddress, fileNames):
    """
        Function to Create the files needed for storing the data
            rootAddress: A string representing the rood directory
            fileNames: List of strings, file names, representing
                the different relationship categories.
        Returns:
            None
    """

    for filename in fileNames:
        filePath = rootAddress + filename + ".json"
        file = open(filePath, "w")
        file.close()


def readData(rootAddress, fileName):
    """
        Function to read the stored data.
        Params:
            rootAddress: String representing the rood directory
            fileName: String representing the file name
              (one for each relationship category)
        Return:
            base: Json representing all the data that been stored
                in the that file
    """

    counter = 0
    base = json.loads("[]")
    filePath = rootAddress + fileName + ".json"
    with open(filePath, mode="r", encoding="utf-8") as feedsjson:
        while True:
            content = feedsjson.readline().strip()
            if content == "":
                break
            base.append(json.loads(content))
            counter += 1

    with open("log.txt", mode="a", encoding="utf-8") as feedsjson:
        feedsjson.write(fileName + "\t" + str(counter) + "\n")
        feedsjson.close()
    return base


def dataPrepration(data):
    """
        Function to prepare a verctorize version of the features
        Params:
            data: Pandas dataframe, representing the tweet's
                text, length, and relationship type category
        Return:
            train_data: Matrix, representing the train dataset
    """

    text = data["text"].map(lambda x: preprocess(x))
    labels = data["hostile"].values
    relationshipDic = {"0": 0, "1": 1, "10": 2, "11": 3}
    results = []
    relationshipType = data["group"].values

    for i in range(len(text)):
        res = {
            "text": text[i],
            "group": relationshipType[i],
            "feature": np.array(
                [relationshipDic[relationshipType[i]], len(text[i])]
            ),  # noqa: E501
            "y": labels[i],
        }
        results.append(res)
    results_df = pd.DataFrame(results)
    featuresString = [
        "relation_%s len_%s" % (x[0], x[1])  # noqa: E501
        for x in results_df["feature"].values
    ]

    vect = feature_extraction.text.CountVectorizer(binary=True, min_df=0.0005)
    features = vect.fit_transform(featuresString).toarray()

    results_df["feature"] = [features[i] for i in range(len(text))]
    train_data = results_df.as_matrix()
    return train_data


def replaceThreeOrMore(word):
    """
        Fucntion to search for 3 or more repetitions of letters
        and replace with this letter itself only once
        Params:
            word: String representing the word in a tweet
        Return:
            A string
    """

    pattern = re.compile(r"(.)\1{3,}", re.DOTALL)
    return pattern.sub(r"\1", word)


def cleanText(text):
    """
        Function to clean the data, removing the stopwords and links,
        replacing the emojis with their string representation, and
        converting all the words to small_case version
        Params:
            text: String representing the text content of a tweet
        Return:
            result: String, representing the cleaned version of
                the text
    """

    text = bytes(text, "utf-8", "ignore").decode("utf-8", "ignore")
    text = demojize(text)  # Add " emoji_" as prefix
    text = re.sub(
        r"#(\w+)", r" hashtag_\1 ", text
    )  # noqa: E501  # Add " hashtag_" as prefix
    text = re.sub(
        "@", " @", text
    )  # noqa: E501  # Substitute mentions with specialmentioned
    text = re.sub(
        "(?:^|[^\w])(?:@)([A-Za-z0-9_](?:(?:[A-Za-z0-9_]|(?:\.(?!\.))){0,28}(?:[A-Za-z0-9_]))?)",  # noqa: E501
        " ",
        text,
    )

    # Substitute urls with specialurl
    text = re.sub("http\S+", " ", text)  # noqa: W605
    text = re.sub(
        " '|'\W|[-(),.!?#*$~`\{\}\[\]/+&*=:\"^]", " ", text  # noqa: W605
    )  # Remove other symbols
    text = re.sub("\s+", " ", text).lower().strip().split()  # noqa: W605
    text_list = [
        replaceThreeOrMore(i) for i in text
    ]  # Remove repetition letters  # noqa: E501
    result = " ".join(text_list)

    if result == " " or result == "":
        return "blank_comment"
    else:
        return result


def preprocess(tweet):
    """
        Functon to do preprocessing cleaning
        Params:
            tweet: String representing the text of a tweet
        Returns:
            text_cleaned: String representing cleaned version
                of the text with no stopword, links, etc
    """

    text = cleanText(tweet).split()
    text_cleaned = " ".join([word for word in text if word not in STOPWORDS])

    return text_cleaned


def auroc(y_true, y_pred):
    """
        Function to Calculate the area under the curve
        Params:
            y_true: List of intigers, representing the true
                label of our data
            y_pred: List of intigers, representing the predicted
                label of our data
        Returns:
            The AUC of the prediction
    """

    try:
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    except ValueError:
        pass

import re
import json
import numpy as np
import pandas as pd
import configparser
from TwitterAPI import TwitterAPI
import sklearn.feature_extraction as feature_extraction

def getTwitter(config_file):
    
    """ 
        Autherizing the twitterAPI access.
        Params:
          config_file....A string representing the config file name
                           , including the consumer_key, consumer_secret,
                           access_token, and access_token_secret
        Returns:
            A Twitetr API handle
    
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    twitter = TwitterAPI(
                   config.get('twitter', 'consumer_key'),
                   config.get('twitter', 'consumer_secret'),
                   config.get('twitter', 'access_token'),
                   config.get('twitter', 'access_token_secret'))
    return twitter

def filePrepration(rootAddress, fileNames):

    """ 
        Creating the files needed for storing the data.
        Params:
          rootAddress...A string representing the rood directory
          fileNames.....A list of file names, representing the
                     different relationship categories.
    """
    
    for filename in fileNames:
        filePath = rootAddress + filename +".json"
        file = open(filePath, 'w')
        file.close()


def readData(rootAddress, fileName):

    """ 
        Reading the stored data.
        Params:
          rootAddress...A string representing the rood directory
          fileName......A string representing the file name
                     (one for each relationship category).
        Return:
          A json file representing all the data that been stored
          in the that file.
    """
    counter = 0
    base = json.loads('[]')
    filePath = rootAddress + fileName + ".json"
    with open(filePath, mode='r', encoding='utf-8') as feedsjson:
        while True:
            content = feedsjson.readline().strip()
            if content == "":
                break
            base.append(json.loads(content))
            counter+=1

    with open('log.txt', mode='a', encoding='utf-8') as feedsjson:
        feedsjson.write(fileName + "    "+ str(counter)+"\n")
        feedsjson.close()
    return base

def dataPrepration(data):
    """ 
        Preparing a verctorize version of the features .
        Params:
          data...A dataframe, representing the tweet's text, length, and relationship type category
          
        Return:
          A matrix, representing the train dataset 
    """
    text = data['text'].map(lambda x: preprocess(x))
    labels = data['hostile'].values
    relationshipDic = {
        '0': 0 ,
        '1': 1,
        '10': 2,
        '11': 3
    }
    results = []
    relationshipType = data['group'].values
    
    for i in range(len(text)):
        res ={
            'text':text[i],
            'group':relationshipType[i],
            'feature': np.array([relationshipDic[relationshipType[i]], len(text[i])]),
            'y':labels[i]
        }
        results.append(res)
    results_df = pd.DataFrame(results)
    featuresString = ['relation_%s len_%s' %
                (x[0], x[1]) for x in results_df['feature'].values]
    
    vect = feature_extraction.text.CountVectorizer(binary = True, min_df = 0.0005)
    features = vect.fit_transform(featuresString).toarray()

    results_df['feature'] = [features[i] for i in range(len(text))]
    train_data  = results_df.as_matrix()
    return train_data

def replaceThreeOrMore(word):
    """ 
        Search for 3 or more repetitions of letters and replace with this letter itself only once 
        Params:
          word...A string representing the word in a tweet 
          
        Return:
          A string
    """

    pattern = re.compile(r"(.)\1{3,}", re.DOTALL)
    return pattern.sub(r"\1", word)


def cleanText(text):
    """ 
        Cleaning the data, removing the stopwords and links, replacing the emojis with their
        string representation, and converting all the words to small_case version 
        Params:
          text...A string representing the text content of a tweet 
          
        Return:
          A string, representing the cleaned version of the text
    """
    text = bytes(text, 'utf-8','ignore').decode('utf-8','ignore')
    # Add " emoji_" as prefix
    text = demojize(text)
    # Add " hashtag_" as prefix
    text = re.sub(r"#(\w+)", r" hashtag_\1 ", text)
    # Substitute mentions with specialmentioned
    text = re.sub('@', ' @', text)
    text = re.sub('(?:^|[^\w])(?:@)([A-Za-z0-9_](?:(?:[A-Za-z0-9_]|(?:\.(?!\.))){0,28}(?:[A-Za-z0-9_]))?)',
                  ' ', text)
    # Substitute urls with specialurl
    text = re.sub('http\S+', ' ', text)
    # Remove other symbols
    text = re.sub(" '|'\W|[-(),.!?#*$~`\{\}\[\]/+&*=:\"^]", " ", text)
    text = re.sub("\s+", " ", text).lower().strip().split()
    # Remove repetition letters
    text_list = [replaceThreeOrMore(i) for i in text]
    
    result = " ".join(text_list)

    if result == " " or result == "":
        return "blank_comment"
    else:
        return result

def preprocess(tweet):
    """
        The cleaning process
        Params:
          tweet....A string representing the text of a tweet
        Returns:
            The cleaned version of the text with no stopword, links, etc 
          
       
    """
    text = cleanText(tweet).split()
    text_cleaned =" ".join([word for word in text if word not in STOPWORDS])
    
    return text_cleaned


def auroc(y_true, y_pred):
    """
        Calculating the area under the curve 
        Params:
          y_true....A list of intigers, representing the true label of our data
          y_pred....A list of intigers, representing the predicted label of our data
        Returns:
            The AUC of the prediction 
          
       
    """
    try:
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    except ValueError:
        pass


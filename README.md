# INFLUENCE OF TIE STRENGTH ON HOSTILITY IN SOCIAL MEDIA
## Problem

In this research, we have studied the influence of the relationship between the conversation participants on both the presence and intensity of hostile comments.

## Research questions

Here are the core questions/subproblems we have addressed:

1. Can hostile words be used in a positive manner? 
2. Do friends use hostile words differently?
3. Can a relationship between the conversation participants change the meaning of conversation?

## Related work

Prior works have studied the users’ behavior in social media in various computational methods. The first set of articles try to predict antisocial behavior in social media outlets or forums, while others make an attempt to find the relation between group norms and communication behavior. A variety of methods have been proposed for cyberbullying and hostility detection. These methods mostly approach the problem by treating it as a classification problem, where the comments been classified as antisocial or not. 

Our approach takes each tweet, extract the participants, their relationship and then extract the meaning of the tweet. This will help us to find which words can be used both as hostile and nonhostile and also what may cause that. For instance, some words, which previously labeled as the top 10 used hostile words on social media, can also be used as an act of surprise or even only for exaggeration purposes. However, one of the biggest challenges here is on how to predict the hostility in the future if we are not aware of the relationship between the participants.

## Data

For data collection, we created the dataset using the Twitter API as the social media of interest. Data is collected from newly posted from January until the end of March 2019. We originally recorded +200,000 tweets which later re-sampled 12,000 tweets to keep the data balanced and be able to label the tweets in the later stages of this research.

To detect hostility in tweets, the content should be parsed and analyzed to find inappropriate words or phrases. We looked for tweets with words that are known for their high level of hostility and then checked whether the relationship type has changed their meaning (in this case, one or both of them unfollow the other user) or they have lowered the hostility in their heated-up discussion.

The first step was to gather a list of hostility representative words which score high on the hostility spectrum. We used a combination of :

  a) Hatebase, an online lexicon of words that are used primarily in Hate Speech
  b) Google list of bad words and top swear words, etc
As a result, we end up with the 65 words such as n-word, f-word, and c-word which then been used as the hostility seed words.

In addition to the tweet content, we also needed to collect the relationship between the original poster and the target exactly after the tweet has been posted because their type of relationship might change after a hostile argument.

The next step was to classify each tweet based on its type of relationship. Twitter is a two-way friendship type of social media, thus, if we assume ”A” write a comment that includes a hostile word in reply to ”B”’s previous comment, then at the time the tweet been posted, they can be classified as one of the following four groups :
  1. Dual friendship: ”A” follows ”B” and ”B” follows ”A” as well. 
  2. No Friendship: Neither ”A” nor ”B” follows the other person. 
  3. One-way Friendship:
    (a) Sender follows Target: Only ”A” follows ”B”.
    (b) Target follows Sender: Only ”B” follows ”A”.

After collecting the tweet, we sampled 3,000 tweets from each type of relationship and merge them together. This results in a huge dataset with 12,000 sampled tweets. The whole idea was to create a large and balanced dataset which then can be employed in our classification models. 

These sampled tweets have then been annotated by three annotators. To this end, annotators first started with annotating the first 500 tweets altogether to get a better understanding of the hostility definition. Then, they annotate over 2000 tweets individually. We also asked them to not label the ambiguous ones that they are not sure about, so the second annotator can go through and label the data. If they disagree on the final label, then we asked the third person to label the data as the tiebreaker. In the end, we ended up with 6.7K of cleaned and relabeled tweets which then can be used in our next steps.


## Methods

The features extracted from data processing were used to construct a model for detecting hostility. We tested several machine learning techniques to select the best classifier. We employed Logistic Regression (LR), Long Short Term Memory (LSTM), and Bi-directional (BD-LSTM).

Our proposal is adding the relationship to the classifier for better hostility detection. we have a BD-LSTM and an LR classifier. Both are using features below:

  • Tweet Context: Exact text or tweet with stopwords removed and emojis replaced
  • Length of Tweet: Size of the tweet in words
  • Relationship Type: The relationship between the sender and the target immediately after the tweet has been posted.

## Results

We used an RNN with bi-directional LSTM and an LR classification model with features including the tweet content, length of the tweet, and the relationship between the sender and the target. Our BD-LSTM model improved the AUC of hostility detection by 4% while providing the F1 of 5% as well. Similarly, the F1-score got increased by 4% using a classifier with LR. We showed that employing the relationship categories as a feature for the classification model results in higher detection accuracy.

![Image](../master/src/figures/AUC.png?raw=true)
![Image](../master/src/figures/accuracy.png?raw=true)

## Conclusions / Future Work

Toxic content in social media is increasing every day and more users are using aggressive, threatening, or bullying language in the heated conversations. We focused on the hostility in social media and proposed a method to detect hostile conversations on Twitter more accurately. While prior research addressed the problem by only analyzing the exact words, we tried to find the actual meaning of the word using the relationship between the participating users. We find that:
  1)Friends (users who follow each other) use profane words mostly a non-hostile behavior to show support or affection. Even if there is hostility within this group, it is frequently towards a third party.
  
  2) Users who do not follow each other tend to engage more in hostile conversations and their aggressiveness is usually directed toward the target.
  
  3) If the target is not following the sender, and there is a hostile conversation happening, usually sender has a huge follower base and might be a celebrity replying to a previous insult.
  
  4) When there is no friendship between users in a hostile discussion, the average length of tweets increases as they show more rage in the ongoing conversation to be able to prove their point of view.
  
  There are several approaches available for future work. First, evaluating our proposed model on a larger dataset and also teach annotators more deeply. Second, inferring more user attributes (e.g. age, gender, ethnicity, geo-location, etc) may provide additional insight for detection. For instance, the N-word is not considered profane in some countries. Third, applying more complex linguistic techniques helps with better sampling (less manual annotation, towards more automation). Using a profane word in a tweet expressing happiness or excitement should not be considered as a possible candidate for a hostile tweet. Finally, use the last 200 tweets of both sender and the target of the tweets to create a conversation chain and use that additional information’s as insight to both detect and also predict hostility in twitter.
  

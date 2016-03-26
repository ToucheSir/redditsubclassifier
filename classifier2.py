from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from csv import DictReader

import numpy as np


def collectComments(file='comments.csv'):
    data = []

    with open(file) as csvdata:
        reader = DictReader(csvdata)
        for row in reader:
            data.append(row['body'])

    return data

def getData():
    left_data = collectComments('political_left.csv')
    right_data = collectComments('political_right.csv')
    leans = np.concatenate((np.zeros(len(left_data)), np.ones(len(right_data))))
    return left_data + right_data, leans

def getVocab():
    tags_right = [
        "obama",
        "government",
        "conservatives",
        "liberals",
        "money",
        "white",
        "republican",
        "black",
        "hillary",
        "media",
        "state",
        "politics",
        "president",
        "tax",
        "gay",
        "police",
        "marriage",
        "democrats",
        "country",
        "bush",
        "pay",
        "win"
        "political",
        "love",
        "hate",
        "law",
        "cruz",
        "clinton",
        "job",
        "racist",
        "rand",
        "republicans",
        "america",
        "wage",
        "god",
        "women",
        "sanders",
        "american",
        "business",
        "gop",
        "election",
        "rights",
        "bernie",
        "democrat",
        "power",
        "war",
        "jobs",
        "islam",
        "religion",
        "taxes",
        "christian",
        "hell",
        "muslims",
        "isis",
        "abortion",
        "science",
        "ted",
        "poor",
        "freedom",
        "congress",
        "college",
        "gun",
        "religious",
        "military",
        "cops",
        "texas",
        "welfare",
        "jesus",
        "guns"
    ]
    tags_left = [
        "socialism",
        "socialist",
        "capitalism",
        "sanders",
        "social",
        "revolution",
        "workers",
        "white",
        "capitalist",
        "socialists",
        "government",
        "bernie",
        "political",
        "liberal",
        "communism",
        "communist",
        "war",
        "money",
        "society",
        "police",
        "comrade",
        "marx",
        "black",
        "politics",
        "american",
        "liberals",
        "democrats",
        "democratic",
        "democracy",
        "power",
        "america",
        "reactionary",
        "free",
        "racist",
        "labour",
        "bourgeois",
        "stalin",
        "union",
        "democrat",
        "racism",
        "ideology",
        "marxist",
        "income",
        "job",
        "labor",
        "human",
        "economic",
        "hillary",
        "unions",
        "lenin",
        "military",
        "their",
        "bourgeoisie",
        "wage",
        "fight",
        "republican",
        "oppression",
        "religion",
        "clinton",
        "obama",
        "revolutionary",
        "president",
        "death",
        "soviet",
        "imperialism",
        "anarchists",
        "solidarity",
        "capitalists",
        "economy",
        "school",
        "media",
        "leftist",
        "radical",
        "propoganda"
    ]
    return list(set(tags_left + tags_right))

if __name__ == '__main__':
    comments, leans = getData()
    vocab = getVocab()
    folds = KFold(len(comments), n_folds=5, shuffle=True)

    for train_i, test_i in folds:
        train_c = [comments[i] for i in train_i]
        train_l = leans[train_i]
        test_c = [comments[i] for i in test_i]
        test_l = leans[test_i]

        text_clf = Pipeline([('vect', CountVectorizer(vocabulary=vocab, binary=True)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])

        text_clf = text_clf.fit(train_c, train_l)

        predicted = text_clf.predict(test_c)

        # for comment, plean, alean in zip(test_c, predicted, test_l):
            # print('%s (%s): %r' % (plean, alean, comment))
        print '\n~~~~~~\n'
        correct = np.count_nonzero(predicted == test_l)
        print correct, len(test_l), float(correct) / float(len(test_l))

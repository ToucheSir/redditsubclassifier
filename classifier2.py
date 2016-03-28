from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed

from csv import DictReader, writer

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier


def collect_comments(filename='comments.csv'):
    data = []

    with open(filename) as csvdata:
        reader = DictReader(csvdata)
        for row in reader:
            data.append(row['body'])

    return data


def get_data():
    left_data = collect_comments('political_left.csv')
    right_data = collect_comments('political_right.csv')
    leans = np.concatenate((np.zeros(len(left_data)), np.ones(len(right_data))))
    return left_data + right_data, leans


def get_vocab():
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


class Runner:
    def __init__(self, comments, leans, vocab=[]):
        self.comments = comments
        self.leans = leans
        self.vocab = vocab

    def run(self, tests=[], trials=5):
        banner = '{}({}): binary={} tfidf={}'
        res = []
        for (clf, args, clf_args) in tests:
            test = [banner.format(clf.__name__, clf_args,
                                  args.get('binary') or False,
                                  args.get('tfidf') or True)]
            test += Parallel(n_jobs=4)(delayed(run_classifier)
                                       (self.comments, self.leans, clf, **args)
                                       for _ in range(trials))
            res.append(test)

        return res


def run_classifier(comments, leans, classifier, args={}, vocab=None, binary=False, tfidf=True, n_folds=10):
    total = 0
    folds = KFold(len(comments), n_folds=n_folds, shuffle=True)

    for train_i, test_i in folds:
        train_c = [comments[i] for i in train_i]
        train_l = leans[train_i]
        test_c = [comments[i] for i in test_i]
        test_l = leans[test_i]

        text_clf = Pipeline([
            ('vect', (TfidfVectorizer if tfidf else
                      CountVectorizer)(vocabulary=vocab, binary=binary)),
            ('clf', classifier(**args)),
        ])
        text_clf = text_clf.fit(train_c, train_l)

        # for comment, plean, alean in zip(test_c, predicted, test_l):
        # print('%s (%s): %r' % (plean, alean, comment))

        predicted = text_clf.predict(test_c)
        correct = np.count_nonzero(predicted == test_l)
        print correct, len(test_l), float(correct) / float(len(test_l))
        total += float(correct) / float(len(test_l))

    return total / n_folds


def write_out(filename, results):
    with open(filename, 'w') as f:
        out = writer(f)
        for row in results:
            out.writerow(row)


if __name__ == '__main__':
    comment_bodies, leanings = get_data()
    vocabulary = get_vocab()
    runner = Runner(comment_bodies, leanings, vocabulary)

    print '''
With Vocab:
~~~~~~~~~~~
    '''
    write_out('no_vocab.csv', runner.run([
        # Decision Trees
        (DecisionTreeClassifier, {}, {}),
        (DecisionTreeClassifier, {'binary': True}, {}),
        # Random Forests
        (RandomForestClassifier, {}, {}),
        (RandomForestClassifier, {'binary': True}, {}),
        (RandomForestClassifier, {}, {'n_estimators': 200}),
        (RandomForestClassifier, {'binary': True}, {'n_estimators': 200}),
        # Naive Bayes
        (MultinomialNB, {}, {}),
        (MultinomialNB, {'binary': True}, {}),
        (BernoulliNB, {}, {}),
        (BernoulliNB, {'binary': True}, {}),
        # Gradient Descent
        (SGDClassifier, {}, {'shuffle': True}),
        (SGDClassifier, {'binary': True}, {'shuffle': True}),
        # SVM (Nu-Support) -> uses linear & rbf kernels
        (NuSVC, {}, {}),
        (NuSVC, {'binary': True}, {}),
        (NuSVC, {}, {'kernel': 'linear'}),
        (NuSVC, {'binary': True}, {'kernel': 'linear'}),
        # SVM (C-Support) -> uses linear & rbf kernels
        (SVC, {}, {}),
        (SVC, {'binary': True}, {}),
        (SVC, {}, {'kernel': 'linear'}),
        (SVC, {'binary': True}, {'kernel': 'linear'}),
        # SVM (linear)
        (LinearSVC, {}, {}),
        (LinearSVC, {'binary': True}, {}),
    ]))

    # TODO(brian): with vocab

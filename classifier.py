from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from csv import DictReader


def collectComments(subs, file='comments.csv'):
    data = []
    targets = []
    mappings = dict((sub, idx) for idx, sub in enumerate(subs))

    with open(file) as csvdata:
        reader = DictReader(csvdata)
        for row in reader:
            if row['subreddit'] in mappings:
                data.append(row['body'])
                targets.append(mappings[row['subreddit']])

    return data, targets


if __name__ == '__main__':
    subs = ['worldnews', 'nottheonion', 'circlejerk', 'atheism', 'gaming']
    data, targets = collectComments(subs)

    tests = ['Florida man', 'fake', 'fps', 'reddit', 'one', 'god']

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    text_clf = text_clf.fit(data, targets)

    predicted = text_clf.predict(tests)

    for doc, category in zip(tests, predicted):
        print('%r => %s' % (doc, subs[category]))

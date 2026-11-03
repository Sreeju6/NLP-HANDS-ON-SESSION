import nltk
import random
from nltk.corpus import names

nltk.download('names', quiet=True)

def gender_features(word):
    return {'last_letter': word[-1].lower()}

labeled_names = (
    [(name, 'male') for name in names.words('male.txt')] +
    [(name, 'female') for name in names.words('female.txt')]
)

random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

train_set, test_set = featuresets[5000:], featuresets[:2000]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(gender_features('David')))
print(classifier.classify(gender_features('Obama')))

print(nltk.classify.accuracy(classifier, test_set))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "Tesseract is a good optical character recognition engine",
    "optical character recognition is significant"
]

vect = CountVectorizer(binary=True)
vect.fit(corpus)

vocab = vect.vocabulary_

for key in sorted(vocab.keys()):
    print("{}:{}".format(key, vocab[key]))

print(vect.transform(["This is a good optical illusion"]).toarray())


doc1 = ["Google Cloud Vision is a character recognition engine"]
doc2 = ["OCR is an optical character recognition engine"]

similarity = cosine_similarity(
    vect.transform(doc1),
    vect.transform(doc2)
)

print(similarity)
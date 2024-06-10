import random
import gensim.models.keyedvectors as word2vec
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine as cosDist
import numpy as np
import json

model = logreg = None

def setModel(Model):
	global model
	model = Model

def findPlagiarism(sentences, target):
	target = target.split(' ')
	def getVector(w):
		if w in model:
			return model[w]
		else:
			return np.zeros(300)
	def sigmoidSimilarity(word1, word2): # returns the similarity of the two words
		t = getVector(word1)
		c = getVector(word2)
		result = 1 / (1 + np.exp(np.dot(-t, c)))
		return result

	sentenceIndex = savedIndex = 0
	maxSimilarity = float('-inf')
	for sentence in sentences:
		index = 0
		similarity = 0
		sentence = sentence.split(' ')
		if len(sentence) > len(target):
			# compare target_sentence to test_sentence word by word
			for word in target:
				simValue = sigmoidSimilarity(word, sentence[index])/len(sentence) # weighted similarity between word1 and word2
				similarity += simValue
				index += 1
			index = len(sentence)-1  # index of last word in test_sentence
			for word in target[::-1]:
				simValue = sigmoidSimilarity(word, sentence[index])/len(sentence) # weighted similarity between word1 and word2
				similarity += simValue
				index -= 1
		else: # test_sentence is less than or equal to target_sentence
			# compare target_sentence to test_sentence word by word
			for word in sentence:
				simValue = sigmoidSimilarity(word, target[index])/len(sentence) # weighted similarity between word1 and word2
				similarity += simValue
				index += 1
			index -= len(target)-1  # index of last word in target_sentence
			for word in sentence[::-1]:
				simValue = sigmoidSimilarity(word, target[index])/len(sentence) # weighted similarity between word1 and word2
				similarity += simValue
				index -= 1
		if similarity > maxSimilarity: # keep test_sentence with most similarity to target_sentence
			maxSimilarity = similarity
			savedIndex = sentenceIndex
		sentenceIndex += 1

	return savedIndex


def classifySubreddit_train(file):
	global logreg

	trainF = open(file, 'r', encoding='utf-8')
	jsonObjects = [json.loads(line) for line in trainF.readlines()]
	data_comments = []
	data_subreddits = []

	def vectorize(sentence):  # turns a sentence into a vector with length 300 (with mean values)
		sentence = sentence.split()
		words_vecs = [model[word] for word in sentence if word in model]
		if len(words_vecs) == 0:
			return np.zeros(300)
		words_vecs = np.array(words_vecs)
		return words_vecs.mean(axis=0)

	for obj in jsonObjects:
		data_comments.append(obj["body"])
		data_subreddits.append(obj["subreddit"])


	X_train = np.array([vectorize(sentence) for sentence in data_comments])  # vectorize each comment from our json object and store in array
	y_train = data_subreddits

	logreg = LogisticRegression(C=1e5, solver='lbfgs', max_iter=500, multi_class='multinomial')
	logreg.fit(X_train, y_train)



def classifySubreddit_test(text):
	log = logreg  # Grab global logistic regression model (should already be fitted)

	def vectorize(sentence):  # turns a sentence into a vector with length 300 (with mean values)
		sentence = sentence.split()
		words_vecs = [model[word] for word in sentence if word in model]
		if len(words_vecs) == 0:
			return np.zeros(300)
		words_vecs = np.array(words_vecs)
		return words_vecs.mean(axis=0)

	text = np.array([vectorize(text)])  # [ [count = 300] ]
	result = log.predict(text)  # =["prediction"]

	return result[0]  # ="prediction"
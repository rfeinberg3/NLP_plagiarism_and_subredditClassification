Based on the instructions provided, here's a README.md file for the repository:

# Plagiarism Detection and Subreddit Classification

## Overview

This repository contains the solution to a set of tasks for detecting plagiarism in sentences and classifying subreddit comments. The implementation is done in Python 3.8+ and utilizes word2vec vectors pre-trained on Google News, logistic regression, and various Python libraries such as gensim, json, sklearn, scipy, and numpy.

## Repository Contents

- `plagiarism_and_subredditClassification.py`: Contains the main functions for the tasks.
- `plagiarism_and_subredditClassification_grader.py`: Grader script to test the implementation.
- `redditComments_train.jsonlist`: Training data for the subreddit classification task.
- `redditComments_test_notGraded.jsonlist`: Test data for evaluating the subreddit classification model.

## Tasks

### Task 1: Plagiarism Detection

**Function:** `findPlagiarism(sentences, target)`

This function identifies which sentence in a list of sentences was most likely plagiarized to create a target sentence. It leverages word2vec vectors to compare the semantic similarity between sentences.

**Input:**
- `sentences`: A list of strings, where each string is a sentence.
- `target`: A string representing a sentence that was plagiarized by rewriting words/phrases to those that are similar.

**Output:**
- Returns the index of the sentence in `sentences` which is most likely to have been used to write `target`.

**Function:** `setModel(model)`

This function is used to set the global word2vec model. It is called once at the beginning of the script to save the model as a global variable and perform any necessary preprocessing.

### Task 2: Subreddit Classification

**Function:** `classifySubreddit_train(trainFile)`

This function trains a logistic regression model to classify comments into subreddits using pre-trained word2vec vectors.

**Input:**
- `trainFile`: The name of a jsonlist file where each line is a JSON object containing a comment and the corresponding subreddit.

**Output:**
- Trains and stores the logistic regression model for subreddit classification.

**Function:** `classifySubreddit_test(comment)`

This function uses the trained logistic regression model to classify a single comment into a subreddit.

**Input:**
- `comment`: A string representing the raw text of a single comment.

**Output:**
- Returns a string which is the name of the subreddit the comment is predicted to belong to.

## Requirements

- Python 3.8+
- gensim
- json
- sklearn
- scipy
- numpy

## Setup

1. Ensure Python 3.8+ is installed on your machine.
2. Install the required libraries using pip:
   ```bash
   pip install gensim json sklearn scipy numpy
   ```
3. Place the `GoogleNews-vectors-negative300.bin` file in the same directory as `plagiarism_and_subredditClassification.py`.

## Usage

1. Set the word2vec model by calling `setModel(model)` at the beginning of your script.
2. Use `findPlagiarism(sentences, target)` to detect plagiarism.
3. Train the subreddit classifier using `classifySubreddit_train(trainFile)`.
4. Classify comments using `classifySubreddit_test(comment)`.

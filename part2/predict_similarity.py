import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logs

import warnings
warnings.filterwarnings("ignore") # suppress warnings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import pickle
import tensorflow.keras.backend as K

# Define constants
MAX_SEQUENCE_LENGTH = 40
TOKENIZER_PATH = 'tokenizer.pickle'
MODEL_PATH = 'malstm_saved_model.keras'

class ExponentNegManhattanDistance(Layer):
    def call(self, inputs):
        left, right = inputs
        return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the entire trained model
malstm = load_model(
    MODEL_PATH,
    custom_objects={"ExponentNegManhattanDistance": ExponentNegManhattanDistance}
)

# Preprocessing function
import re
from bs4 import BeautifulSoup

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet

# Set the NLTK data path to the directory where WordNet is located
nltk.data.path.append('myenv\nltk_data\corpora\wordnet')
nltk.data.path.append('myenv\nltk_data\corpora\stopwords')
nltk.data.path.append('myenv\nltk_data\taggers\averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
STOP_WORDS = stopwords.words("english")

def get_wordnet_pos(treebank_tag):
    """Map NLTK's POS tags to WordNet's POS tags for lemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }
    
    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q, features="html.parser")
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    # Tokenize and remove stop words
    tokens = q.split()
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]

    # POS tagging and Lemmatization
    pos_tagged = pos_tag(filtered_tokens)
    q = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tagged])
    
    return q

# Function for making predictions
def predict_similarity(question1, question2):
    # Preprocess the questions
    q1 = preprocess(question1)
    q2 = preprocess(question2)

    # Convert to sequences using the tokenizer
    seq1 = tokenizer.texts_to_sequences([q1])
    seq2 = tokenizer.texts_to_sequences([q2])

    # Pad the sequences
    data_1 = pad_sequences(seq1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(seq2, maxlen=MAX_SEQUENCE_LENGTH)

    # Predict similarity score
    similarity_score = malstm.predict([data_1, data_2]) # batch_size = 1
    return similarity_score[0][0]  # Return the similarity score

# Example usage
if __name__ == "__main__":
    question1 = input("Enter the first question: ")
    question2 = input("Enter the second question: ")
    
    score = predict_similarity(question1, question2)
    print(f"Similarity score between the two questions: {score:.4f}")
    
    flag = score>=0.5
    if flag:
        print(f'The questions: "{question1}" and "{question2}" are similar.')
    else:
        print(f'The questions: "{question1}" and "{question2}" are not similar.')
# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Importing the NLP libraries
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('webtext')
nltk.download('treebank')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.book import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as et
from xml.sax.saxutils import escape, unescape

def XMLtoString(file):
    with open(file) as data:
        contents = data.read()
        regex = re.compile(r"&(?!amp;)")
        myxml = regex.sub("&amp;", contents)
    return myxml

j=0
for i in os.listdir('./Dataset/The Blog Authorship Dataset'):
	print("File : " + i)

	path = r'./Dataset/The Blog Authorship Dataset/' + i
	contents = XMLtoString(path)
	regex = re.compile(r"&(?!amp;)")	
	myxml = regex.sub("&amp;", contents)

	root = et.fromstring(myxml)

	dates = []
	posts = []
	for blogpost in root:
	    if(blogpost.tag == 'date'):
	        dates.append(blogpost.text)
	    if(blogpost.tag == 'post'):
	        # Converting data to lowercase before inserting in dataframe
	        post = blogpost.text
	        post = post.lower()
	        posts.append(post)

	df = pd.DataFrame()
	df['dates'] = dates
	df['posts'] = posts

	# Sentence Tokenization
	blog_sentences = []
	for post in df['posts']:
	    sentences = sent_tokenize(post)
	    blog_sentences.append(sentences)
	df['sentence_tokenize'] = blog_sentences

	# Saving sentence tokinization to file
	path = r'C:\Users\shrea\Desktop\Jupyter Notebooks\IITB Internship\IITB-ML-Assignment\Output\NLP\\'
	df.to_csv(path+str(j)+'_sentence.csv', index=False)
	print('Saved file to disk.')

	# Word Tokenization

	word_lists = []
	for sentences in df['sentence_tokenize']:
	    for sentence in sentences:
	        words = word_tokenize(sentence)
	    word_lists.append(words)
	df['word_tokenize'] = word_lists

	# Saving word tokenization to file
	df.to_csv(path+str(j)+'_word.csv', index=False)
	print('Saved file to disk.')

	for word_list in word_lists:
	    freq = FreqDist(word_list)
	    # freq.plot(10)

	freq = FreqDist(word_list)
	freq_dict = dict(freq)
	freq_words = list(freq_dict.keys())
	frequencies = list(freq_dict.values())
	freq_df = pd.DataFrame(list(zip(freq_words, frequencies)), columns=['Word','Freq'])
	freq_df.to_csv(path+str(j)+'_frequency.csv', index=False)
	print('Saved file to disk.')

	# Stopwords and Non Stopwords

	stop = []
	non_stopwords = []
	for word_list in word_lists: 
	    for word in word_list:
	        if not word in set(stopwords.words('english')):
	            non_stopwords.append(word)
	        else:
	            stop.append(word)

	stop_df = pd.DataFrame(stop, columns=['Stopwords'])
	# Saving stopwords to file
	stop_df.to_csv(path+str(j)+'_stopwords.csv', index=False)
	print('Saved file to disk.')

	non_stop_df = pd.DataFrame(non_stopwords, columns=['Non Stopwords'])
	# Saving non stopwords to file
	non_stop_df.to_csv(path+str(j)+'_non_stopwords.csv', index=False)
	print('Saved file to disk.')

	# Lexicon Normalization

	# Stemming
	ps = PorterStemmer()
	stem = [ps.stem(word) for word in non_stopwords]
	stem_df = pd.DataFrame(stem, columns=['Stemmed Words'])
	# Saving word stemmed words to file
	stem_df.to_csv(path+str(j)+'_stems.csv', index=False)
	print('Saved file to disk.')

	# Lemmatization
	lemmatizer = WordNetLemmatizer() 
	lemma = [lemmatizer.lemmatize(word) for word in non_stopwords]
	lemma_df = pd.DataFrame(lemma, columns=['Lemma'])
	# Saving word stemmed words to file
	lemma_df.to_csv(path+str(j)+'_lemmas.csv', index=False)
	print('Saved file to disk.')

	j+=1
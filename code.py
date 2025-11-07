!pip install pydub
!pip install spleeter

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import os

#STEP 1 -> AUDIO TRIMMING

# Define the input and output directories
input_dir = "/Users/dhwanibhavankar/Downloads/911_recordings"
output_dir = "/Users/dhwanibhavankar/Desktop/911_trimmed"

# Create the output directory 
os.makedirs(output_dir, exist_ok=True)

# Iterate through the audio files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".mp3"):  # Adjust the file extension if necessary
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_trimmed.mp3")

        try:
            # Load the audio file
            audio = AudioSegment.from_file(input_path)

            # Trim the audio to 2 minutes (120,000 milliseconds)
            trimmed_audio = audio[:120000]

            # Export the trimmed audio to the output directory
            trimmed_audio.export(output_path, format="mp3")  
            print(f"Trimmed and saved: {output_path}")
        except CouldntDecodeError as e:
            print(f"Could not decode {input_path}: {e}")
            continue  # Skip this file and continue with the next one


#STEP 2 -> AUDIO ENHANCING

from spleeter.separator import Separator

# Define the input and output directories
input_directory = "/content/drive/MyDrive/EDA_DPA/Calls_Audio/911_trimmed"
output_directory = "/content/drive/MyDrive/EDA_DPA/Calls_Audio/911_vocals"


os.makedirs(output_directory, exist_ok=True)

# Initialize the Spleeter separator
separator = Separator('spleeter:2stems')

# Loop through all audio files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".mp3"):  
        input_path = os.path.join(input_directory, filename)

        # Separate vocals and accompaniment
        separator.separate_to_file(input_path, output_directory)

        # Rename the separated vocal file to the original filename
        vocals_file = os.path.join(output_directory, f"{filename[:-4]}_vocals.mp3")
        output_path = os.path.join(output_directory, filename)

        # Rename the file 
        if os.path.exists(vocals_file):
            os.rename(vocals_file, output_path)
            print(f"Extracted vocals from {filename} and saved to {output_path}")
        else:
            print(f"Failed to extract vocals from {filename}")


#STEP 3 -> FEATURE EXTRACTION

import os
import glob
import librosa
import numpy as np
import pandas as pd

# Function to extract MFCC features
def mfcc_extractor(audio):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Function to extract zero-crossing rate features
def zero_extractor(audio):
    zeros = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512, center=True)
    zeros_scaled_features = np.mean(zeros.T, axis=0)
    return zeros_scaled_features

# Function to extract RMS features
def rms_extractor(audio):
    rms = librosa.feature.rms(y=audio)
    rms_scaled_features = np.mean(rms.T, axis=0)
    return rms_scaled_features

# Function to extract spectral centroid features
def spectral_centroid_extractor(audio):
    sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    sc_scaled_features = np.mean(sc.T, axis=0)
    return sc_scaled_features

# Function to extract spectral bandwidth features
def spectral_bandwidth_extractor(audio):
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    sb_scaled_features = np.mean(sb.T, axis=0)
    return sb_scaled_features

# Function to extract spectral contrast features
def spectral_contrast_extractor(audio):
    sco = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    sco_scaled_features = np.mean(sco.T, axis=0)
    return sco_scaled_features

# Function to extract polynomial features
def polynomial_extractor(audio):
    poly = librosa.feature.poly_features(y=audio, sr=sample_rate, order=2)
    poly_scaled_features = np.mean(poly.T, axis=0)
    return poly_scaled_features

# Function to recursively search for vocal audio files in subdirectories
def process_subdirectories(root_directory):
    df = pd.DataFrame(columns=["File_Path", "mfcc", "zero_features", "rms_features", "sc_features", "sb_features", "sco_features", "poly_features"])

    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".wav"):
                vocal_audio_file_path = os.path.join(root, file)
                vocal_audio, sample_rate = librosa.load(vocal_audio_file_path)
                mfcc = mfcc_extractor(vocal_audio)
                zero = zero_extractor(vocal_audio)
                rms = rms_extractor(vocal_audio)
                spectral_centroid = spectral_centroid_extractor(vocal_audio)
                spectral_bandwidth = spectral_bandwidth_extractor(vocal_audio)
                spectral_contrast = spectral_contrast_extractor(vocal_audio)
                polynomial = polynomial_extractor(vocal_audio)
                df = df.append({"File_Path": vocal_audio_file_path, "mfcc": mfcc, "zero_features": zero,
                                "rms_features": rms, "sc_features": spectral_centroid, "sb_features": spectral_bandwidth,
                                "sco_features": spectral_contrast, "poly_features": polynomial}, ignore_index=True)

    return df

# Directory where our vocal audio files are stored
vocals_directory = "/content/drive/MyDrive/EDA_DPA/Calls_Audio/911_vocals"
result_df = process_subdirectories(vocals_directory)
result_df.to_csv("Features.csv")

data_df = pd.read_csv('/content/drive/MyDrive/EDA_DPA/Calls_Audio/Features/metadata.csv')
features_df = pd.read_csv('/content/drive/MyDrive/EDA_DPA/Calls_Audio/Features/Features.csv')

final_df = pd.concat([data_df, features_df], axis=1)
final_df = pd.DataFrame(final_df)

def calculate_median(arr):
    return np.median(arr)

# Apply the calculate_median function to each cell in the 'mfcc' column
features['mfcc'] = features['mfcc'].apply(calculate_median)
features['sco_features'] = features['sco_features'].apply(calculate_median)
features['poly_features'] = features['poly_features'].apply(calculate_median)


#STEP 4 -> DATA CLEANING

import re

def label_title(row):
    # Define a list of keywords and their variations
    keywords = ['accident', 'prank', 'fake', 'date', 'Non-emerg']

    # Create a regex pattern to match any form of the keywords
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in keywords) + r')\b'

    # Check if the title contains the pattern or if false_alarm is 1
    if re.search(pattern, row.title, flags=re.IGNORECASE) or row.false_alarm == 1:
        return 'prank'
    return 'genuine'

# Apply the label_title function to create a new 'label' column
df['label'] = df.apply(label_title, axis=1)

df.isna().sum() #state, false_alarm and potential_death columns had NaN values
majority_state = df['state'].mode()[0]
df['state'].fillna(majority_state, inplace=True)
df['false_alarm'] = df.apply(lambda row: 1 if row['label'] == 'prank' else 0 if row['label'] == 'genuine' else row['false_alarm'], axis=1)

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=1)
# Define the column to be imputed
impute_column = ["potential_death"]

# Fit and transform the imputer on the specified column
df[impute_column] = imputer.fit_transform(df[impute_column])


#STEP 5 -> FINDING CORRELATIONS 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Now the 'label_encoded' column will contain 0 for 'genuine' and 1 for 'prank'

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# WE FOUND THAT THE CORRELATION BETWEEN ALL WERE VERY LESS
# SO TO CHECK THE EXISTANCE OF NON-LINEAR RELATIONS WE TRIED APPLYING RANDOM FORREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['mfcc', 'zero_features', 'rms_features','sc_features','sb_features','sco_features','poly_features']], df['label_encoded'], test_size=0.2, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) #GAVE AN ACCURACY OF 96% BUT THE CONFUSION MATRIX SHOWED THAT THE MODEL PREDICTED ALL THE CALLS TO GENUINE

#STEP 6 -> ANAMOLY DETECTION

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("/content/drive/MyDrive/EDA_DPA/Features/final_df.csv")

# Assuming 'mfcc', 'zero_features', 'rms_features', 'sc_features', 'sb_features',
# 'sco_features', and 'poly_features' are the features you want to use for anomaly detection

# Selecting the features we want to use for anomaly detection
X = data[['mfcc', 'zero_features', 'rms_features', 'sc_features', 'sb_features', 'sco_features', 'poly_features']]

# Create an Isolation Forest model
clf = IsolationForest(contamination=0.07, random_state=42)  # Adjust the contamination parameter based on the expected anomaly rate

# Fit the model to the data
clf.fit(X)

# Predict anomalies (1 for normal, -1 for anomalies)
predictions = clf.predict(X)

# Add the anomaly predictions to the original dataset
data['anomaly'] = predictions

# Evaluate the model using precision, recall, and F1-score
precision = precision_score(data['anomaly'], predictions, pos_label=-1)
recall = recall_score(data['anomaly'], predictions, pos_label=-1)
f1 = f1_score(data['anomaly'], predictions, pos_label=-1)

# Calculate the average precision score (AUC-PR)
average_precision = average_precision_score(data['anomaly'], predictions)

# Print the evaluation metrics
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-Score: {:.2f}".format(f1))
print("Average Precision: {:.2f}".format(average_precision))

# Create a scatter plot to visualize anomalies
plt.figure(figsize=(10, 6))
plt.scatter(X.index, X['mfcc'], c=predictions, cmap='viridis')
plt.title('Anomaly Detection: MFCC Feature')
plt.xlabel('Sample Index')
plt.ylabel('MFCC Value')
plt.show()

#THOUGH THE ACCURACY PRECISION RECALL ALL CAME OUT TO BE 1.00 SHOWING OVERFITTED MODEL.

#DIVERSE THE PERSPECTIVE
#STEP 7 -> KEY WORDS EXTRACTION

import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize the text
    tokens = text.lower().split()
    # Remove punctuation and numbers
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stopwords
    stop = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Apply the preprocessing to our DataFrame
df['title_tokens'] = df['title'].apply(preprocess_text)

# Create a dictionary
dictionary = corpora.Dictionary(df['title_tokens'])

# Create a document-term matrix
doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in df['title_tokens']]

# Specify the number of topics you want to discover
num_topics = 7

# Create the LDA model
lda_model = gensim.models.LdaModel(
    doc_term_matrix,
    num_topics=num_topics,
    id2word=dictionary,
    passes=15
)

# Print the topics and their top words
for topic in lda_model.print_topics():
    print(topic)

#THOUGH IT CREATED DIFFERENT TOPICS, IT HAD WORDS LIKE THE, IN AND...., SO WE TRIED USING A LIBRARY CALLED LATENTDIRICHLETALLOCATION
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Define custom stopwords
custom_stopwords = ["the", "and", "to", "in", "be", "911"]

# TF-IDF Vectorization with custom stopwords
tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stopwords, min_df=2)
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['description'])

# number of topics
num_topics = 10  
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tfidf_matrix)

# Display the top words for each topic
top_words = get_top_words(lda, tfidf_vectorizer)

for i, words in enumerate(top_words):
    print(f"Topic {i + 1}: {', '.join(words)}")
#THOUGH THIS CREATED TOPICS WITH IDEAL KEYWORDS THERE WAS AN ISSUE OF REPEATITION AND SIMILARITY WITHIN THE TOPICS


#STEP 8 -> SENTIMENT ANALYSIS (THE FINAL ONE THAT WORKED)

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')  # Download the VADER lexicon
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to the 'description' column
new_df['sentiment'] = new_df['description'].apply(analyze_sentiment)

new_df['sentiment'].value_counts()
# Negative    580 | Positive     79 | Neutral      51

pip install scikit-learn nltk

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

new_df['compound_score'] = new_df['description'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Split the data into training and testing sets
X = new_df['description']
y = new_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust the number of features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

#THIS WORKED AND GAVE AN ACCURACY OF 87%.





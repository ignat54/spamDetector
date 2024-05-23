from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import string

nltk.download('stopwords')
ps = nltk.PorterStemmer()
data = pd.read_csv("SMSSpamCollection", sep='\t')
data.columns = ['label', 'body_text']


def count_punct(text):
    if len(text) - text.count(" ") == 0:
        return 0
    else:
        count = sum([1 for char in text if char in string.punctuation])
        return round(count/(len(text) - text.count(" ")), 3) * 100


data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))


def clean_text(text):
    text = "".join([word.lower()
                   for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


def load_model():
    rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)
    model = rf.fit(X_train_vect, y_train)
    return model


def clean_text(text):
    text = "".join([word.lower()
                   for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


tfidf_vect = TfidfVectorizer(analyzer=clean_text, max_features=500)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf_feat = pd.concat(
    [data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
X_features = pd.concat([data['body_len'], data['punct%'],
                       pd.DataFrame(X_tfidf.toarray())], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    data[['body_text', 'body_len', 'punct%']], data['label'], test_size=0.2)
tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])
tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])
X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True),
                         pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True),
                        pd.DataFrame(tfidf_test.toarray())], axis=1)
X_train_vect.columns = X_train_vect.columns.astype(str)
X_test_vect.columns = X_test_vect.columns.astype(str)

st.title("Spam Detection App")

user_input = st.text_input("Enter a message:")
input_data = pd.DataFrame({'body_text': [user_input], 'body_len': [len(
    user_input) - user_input.count(" ")], 'punct%': [count_punct(user_input)]})
input_tfidf = tfidf_vect_fit.transform(input_data['body_text'])
input_vect = pd.concat([input_data[['body_len', 'punct%']].reset_index(
    drop=True), pd.DataFrame(input_tfidf.toarray())], axis=1)
input_vect.columns = input_vect.columns.astype(str)

ps = PorterStemmer()

model = load_model()
prediction = model.predict(input_vect)
if st.button("Predict"):
    st.write(f"Message: {user_input}")
    st.write(f"Prediction: {prediction[0]}")

    y_pred = model.predict(X_test_vect)
    jaccard = jaccard_score(y_test, y_pred, pos_label='spam')
    st.write(f"Jaccard Score: {round(jaccard, 3)}")

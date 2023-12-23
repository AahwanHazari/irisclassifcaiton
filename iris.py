import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


iris_df = pd.read_csv("iris-species.csv")


iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})


X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

rfc_model = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rfc_model.fit(X_train, y_train)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

@st.cache()
def prediction(model, SepalLength, SepalWidth, PetalLength, PetalWidth):
  species = model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

st.sidebar.title("Iris Flower Species Prediction App")

s_length = st.sidebar.slider("Sepal Length", float(iris_df['SepalLengthCm'].min()), float(iris_df['SepalLengthCm'].max()))
s_width = st.sidebar.slider("Sepal Width", float(iris_df['SepalWidthCm'].min()), float(iris_df['SepalWidthCm'].max()))
p_length = st.sidebar.slider("Petal Length", float(iris_df['PetalLengthCm'].min()), float(iris_df['PetalLengthCm'].max()))
p_width = st.sidebar.slider("Petal Width", float(iris_df['PetalWidthCm'].min()), float(iris_df['PetalWidthCm'].max()))

classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if st.sidebar.button('predict'):
  if classifier == 'Support Vector Machine':
    species_type = prediction(svc_model, s_length, s_width, p_length, p_width)
    score = svc_model.score(X_train, y_train)

  elif classifier == 'Logistic Regression':
    species_type = prediction(lr_model, s_length, s_width, p_length, p_width)
    score = lr_model.score(X_train, y_train)

  else:
    species_type = prediction(rfc_model, s_length, s_width, p_length, p_width)
    score = rfc_model.score(X_train, y_train)

  st.write('Species Predicted: ', species_type)
  st.write('Accuracy of the Model is: ', score)

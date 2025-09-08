import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
import pickle   # ✅ NEW: for saving/loading model

# -------------------------------------------------
# Load dataset
df = pd.read_csv("feeds.csv")

# Removing Null values (hardcoded rows dropped)
update_df = df.drop([199, 200, 201, 202])

# Features
x = update_df.iloc[:, [0,1,2,3,4]].values

# KMeans clustering (not directly used for prediction, but kept)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
y = kmeans.fit_predict(x)

# -------------------------------------------------
# ✅ Check if pickle file exists, else train and save
try:
    with open("motor_model.pkl", "rb") as f:
        classifier = pickle.load(f)
        print("✅ Model loaded from pickle file.")
except FileNotFoundError:
    print("⚡ Training model since pickle not found...")
    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
    classifier.fit(x, y)

    # Save trained model
    with open("motor_model.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print("💾 Model trained and saved as motor_model.pkl")

# -------------------------------------------------
# Prediction function
def give_pred(test):
    prediction = classifier.predict(test)
    if prediction == 0:
        return "Your System Failed"
    else:
        return "Your System Works"

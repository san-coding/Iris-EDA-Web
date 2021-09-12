from typing import List
from matplotlib import colors
import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Explore Iris Dataset")


st.write(
    """
### with Sandeep Rajakrishnan ([Github](https://github.com/san-coding))

# Explore different Classifiers
Which one is the best ?
"""
)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris Dataset",))

classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Logistic Regression", "KNN", "Random Forest", "SVM")
)
data = pd.read_csv("Iris.csv")


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = pd.read_csv("Iris.csv")
    else:
        data = pd.read_csv("Iris.csv")
    X = data.drop("Species", axis=1)
    y = data.Species
    return X, y


X, y = get_dataset(dataset_name)
st.write("**Shape of dataset** : ", X.shape)
st.write("**Number of classes** : ", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K Value", 1, 15)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("C Value", 0.01, 10.0)
        params["C"] = C

    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("Maximum Depth (max_depth)", 2, 15)
        n_estimators = st.sidebar.slider("Number of Estimators (n_estimators)", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    elif clf_name == "Logistic Regression":
        penalty = st.sidebar.selectbox(
            "Select Regualrization (penalty)", ("none", "l2")
        )
        params["penalty"] = penalty

    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):

    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(C=params["C"])

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=1234,
        )

    elif clf_name == "Logistic Regression":
        clf = LogisticRegression(penalty=params["penalty"], random_state=1234)

    return clf


clf = get_classifier(classifier_name, params)

# Classification

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf.fit(X_train.drop("Id", axis=1), y_train)

y_pred = clf.predict(X_test.drop("Id", axis=1))

accuracy = accuracy_score(y_test, y_pred)

st.write("**Classifier** : ", classifier_name)

st.write(f"**Accuracy** = {accuracy}")

st.write(
    """
## Lets do some predictions

"""
)
with st.form(key="PredictForm"):
    col1, col2, col3, col4 = st.columns([4, 3, 2, 1])
    with col1:
        sepal_length = st.number_input("Enter Sepal Length : ")
    with col2:
        sepal_width = st.number_input("Enter Sepal Width : ")
    with col1:
        petal_length = st.number_input("Enter Petal Length : ")
    with col2:
        petal_width = st.number_input("Enter Petal Width : ")
    with col1:
        predict_btn = st.form_submit_button(label="Predict")

if predict_btn:
    list = [
        {
            "SepalLengthCm": sepal_length,
            "SepalWidthCm": sepal_width,
            "PetalLengthCm": petal_length,
            "PetalWidthCm": petal_width,
        }
    ]
    predict_df = pd.DataFrame(list)
    species = clf.predict(predict_df)
    st.write(f"**Predicted species** :  {species[0]}")

st.write(
    """
## What does the data look like ?

"""
)
st.table(data.sample(5).drop("Id", axis=1))


# PLOT
st.write(
    """ 
# Data Visualisation

"""
)

st.write(
    """ 
## Scatter Plots

"""
)

st.set_option("deprecation.showPyplotGlobalUse", False)
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
cm = plt.cm.get_cmap("RdYlBu")
xy = range(150)
z = xy
plt.scatter(x1, x2, c=z, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Scatter plot plotted after applying PCA")
plt.colorbar()


st.pyplot(fig)


data.plot(
    kind="scatter", x="SepalLengthCm", y="SepalWidthCm", c=z, alpha=0.8, cmap="viridis"
)
plt.title("Sepal Length vs  Sepal Width")

plt.show()
st.pyplot()

# Modify the graph above by assigning each species an individual color.
sns.FacetGrid(data, hue="Species", size=5).map(
    plt.scatter, "SepalLengthCm", "SepalWidthCm"
).add_legend()
plt.title("Sepal Length vs  Sepal Width with Hue")

plt.show()
st.pyplot()

sns.FacetGrid(data, hue="Species", size=5).map(
    plt.scatter, "PetalLengthCm", "PetalWidthCm"
).add_legend()
plt.title("Petal Length vs  Petal Width with Hue")

plt.show()
st.pyplot()

st.write(
    """ 
## Box Plots to detect outliers

"""
)

ax = sns.boxplot(x="Species", y="PetalLengthCm", data=data)
ax = sns.stripplot(
    x="Species", y="PetalLengthCm", data=data, jitter=True, edgecolor="gray"
)
plt.title("Box plots for Petal Lengths")
plt.show()
st.pyplot()

ax = sns.boxplot(x="Species", y="PetalWidthCm", data=data)
ax = sns.stripplot(
    x="Species", y="PetalWidthCm", data=data, jitter=True, edgecolor="gray"
)
plt.title("Box plots for Petal Widths")
plt.show()
st.pyplot()

ax = sns.boxplot(x="Species", y="SepalLengthCm", data=data)
ax = sns.stripplot(
    x="Species", y="SepalLengthCm", data=data, jitter=True, edgecolor="gray"
)
plt.title("Box plots for Sepal Lengths")
plt.show()
st.pyplot()

ax = sns.boxplot(x="Species", y="SepalWidthCm", data=data)
ax = sns.stripplot(
    x="Species", y="SepalWidthCm", data=data, jitter=True, edgecolor="gray"
)
plt.title("Box plots for Sepal Widths")
plt.show()
st.pyplot()

data.drop("Id", axis=1).boxplot(by="Species", figsize=(10, 10))
plt.title("Box plots grouped by Species")
plt.show()
st.pyplot()


st.write(
    """ 
## Violin Plots
"""
)

sns.violinplot(x="Species", y="PetalLengthCm", data=data, size=6)
plt.title("Violin plot for Petal Length")
plt.show()
st.pyplot()

sns.violinplot(x="Species", y="PetalWidthCm", data=data, size=6)
plt.title("Violin plot for Petal Width")
plt.show()
st.pyplot()

sns.violinplot(x="Species", y="SepalLengthCm", data=data, size=6)
plt.title("Violin plot for Sepal Length")
plt.show()
st.pyplot()

sns.violinplot(x="Species", y="SepalWidthCm", data=data, size=6)
plt.title("Violin plot for Sepal Width")
plt.show()
st.pyplot()

st.write(
    """ 
## Pair Plot
"""
)

sns.pairplot(data.drop("Id", axis=1), hue="Species", size=3)
plt.title("Pair Plots")

plt.show()
st.pyplot()

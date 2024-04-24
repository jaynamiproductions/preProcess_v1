from flask import Blueprint, render_template, request, redirect
import pandas as pd
import numpy as np
from testing.preprocess import PreProcess, Scale, processNew

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

views = Blueprint('views',__name__)

### Split data ###
raw = pd.read_csv('testing/diabetic_data.csv')
df = PreProcess(raw).processed()
train, test = np.split(df.sample(frac=1), [int(0.75*len(df))])
train, X_train, y_train = Scale(train, scale=False, oversample=False)
test, X_test, y_test = Scale(test, scale=False, oversample=False)
###

@views.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@views.route('/models',methods=['GET', 'POST'])
def models():
    if request.method == 'POST':
        model = request.form.get('model')

        if model == 'Logistic Regression':
            return redirect('/logistic-regression')

        elif model == 'K-Nearest Neighbors':
            return redirect('/k-nearest-neighbors')

        elif model == 'Decision Tree Classifier':
            return redirect('/decision-tree')

        elif model == 'Random Forest Classifier':
            return redirect('/random-forest')
        else:
            return render_template('models.html', msg='No model selected. Please make a selection.')     
    else:
        return render_template('models.html')
    
@views.route('/logistic-regression',methods=['GET', 'POST'])
def log_reg():
    log_model = LogisticRegression(max_iter=5000)
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return render_template('logreg.html', score=round(score,4))

@views.route('/k-nearest-neighbors',methods=['GET', 'POST'])
def knn():
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return render_template('knn.html', score=round(score,4))

@views.route('/decision-tree',methods=['GET', 'POST'])
def dtc():
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return render_template('dtc.html', score=round(score,4))

@views.route('/random-forest',methods=['GET', 'POST'])
def rfc():
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return render_template('rfc.html', score=round(score,4))
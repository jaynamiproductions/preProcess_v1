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
        return render_template('models.html')
    
def get_info():
    new = {
        'Other': 0,
        'Caucasian': 0,
        'AfricanAmerican': 0,
        'Asian': 0,
        'Hispanic': 0,
        'gender': '',
        'age': '',
        'time_in_hospital': 0,
        'num_lab_procedures': 0,
        'num_procedures': 0,
        'num_medications': 0,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 0,	
        'number_diagnoses': 0,	
        'diabetesMed': ''
    }
    name = request.form.get('name')
    race = request.form.get('race')
    gender = request.form.get('gender')
    age = request.form.get('age')
    time_in_hospital = request.form.get('time_in_hospital')
    num_lab_procedures = request.form.get('num_lab_procedures')
    num_procedures = request.form.get('num_procedures')
    num_medications = request.form.get('num_medications')
    number_outpatient = request.form.get('number_outpatient')
    number_emergency = request.form.get('number_emergency')
    number_inpatient = request.form.get('number_inpatient')
    number_diagnoses = request.form.get('number_diagnoses')
    med = request.form.get('med')

    new[race] = 1
    new['gender'] = gender
    new['age'] = age
    new['time_in_hospital'] = int(time_in_hospital)
    new['num_lab_procedures'] = int(num_lab_procedures)
    new['num_procedures'] = int(num_procedures)
    new['num_medications'] = int(num_medications)
    new['number_outpatient'] = int(number_outpatient)
    new['number_emergency'] = int(number_emergency)
    new['number_inpatient'] = int(number_inpatient)
    new['number_diagnoses'] = int(number_diagnoses)
    new['diabetesMed'] = med
    return name, new

@views.route('/logistic-regression',methods=['GET', 'POST'])
def log_reg():
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    if request.method == 'POST':
        name, new = get_info()
        new = processNew(new)
        final_pred = log_model.predict(new)
        if final_pred[0] == 0:
            pred = 'will NOT be readmitted within 30 days.'
        else:
            pred = 'will be readmitted within 30 days.'
        return render_template('logreg.html', score=round(score,4), name=name, prediction=pred)
    else:
        return render_template('logreg.html', score=round(score,4))

@views.route('/k-nearest-neighbors',methods=['GET', 'POST'])
def knn():
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    if request.method == 'POST':
        name, new = get_info()
        new = processNew(new)
        final_pred = knn_model.predict(new)
        if final_pred[0] == 0:
            pred = 'will NOT be readmitted within 30 days.'
        else:
            pred = 'will be readmitted within 30 days.'
        return render_template('knn.html', score=round(score,4), name=name, prediction=pred)
    else:
        return render_template('knn.html', score=round(score,4))

@views.route('/decision-tree',methods=['GET', 'POST'])
def dtc():
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    if request.method == 'POST':
        name, new = get_info()
        new = processNew(new)
        final_pred = dtc.predict(new)
        if final_pred[0] == 0:
            pred = 'will NOT be readmitted within 30 days.'
        else:
            pred = 'will be readmitted within 30 days.'
        return render_template('dtc.html', score=round(score,4), name=name, prediction=pred)
    else:
        return render_template('dtc.html', score=round(score,4))

@views.route('/random-forest',methods=['GET', 'POST'])
def rfc():
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    if request.method == 'POST':
        name, new = get_info()
        new = processNew(new)
        final_pred = rf.predict(new)
        if final_pred[0] == 0:
            pred = 'will NOT be readmitted within 30 days.'
        else:
            pred = 'will be readmitted within 30 days.'
        return render_template('rfc.html', score=round(score,4), name=name, prediction=pred)
    else:
        return render_template('rfc.html', score=round(score,4))
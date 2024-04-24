import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

class PreProcess:
    def __init__(self, df):
        self.df = df

    def columnFilter(self):
        cols = ['race', 'gender', 'age', 'time_in_hospital', 
                'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 
                'number_diagnoses', 'diabetesMed', 'readmitted']
        df = self.df[cols]
        return df

    def textCleaner(self):
        df = self.columnFilter()
        df['age'] = df['age'].str.replace(')',']')
        return df
    
    def noneDropper(self):
        df = self.textCleaner()
        cond = df['race'] != '?'
        cond1 = df['gender'] == 'Male'
        cond2 = df['gender'] == 'Female'
        return df[cond][cond1|cond2]
    
    def ordinalMapper(self):
        df = self.noneDropper()
        df['gender'] = df['gender'].map({'Male': 0,'Female': 1})
        df['diabetesMed'] = df['diabetesMed'].map({'No': 0,'Yes': 1})
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 0, '<30': 1})
        df['age'] = df['age'].map({'[0-10]': 1, '[10-20]': 2, '[20-30]': 3, '[30-40]': 4,'[40-50]': 5, '[50-60]': 6, '[60-70]': 7, '[70-80]': 8, '[80-90]': 9, '[90-100]': 10})
        return df
    
    def featureEncoder(self):
        df = self.ordinalMapper()
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(df[['race']]).toarray()

        vals = list(set(df['race'].values))

        for i in range(len(matrix.T)):
            df.insert(0, vals[i], matrix.T[i])

        return df.drop(['race'], axis=1)
    
    def processed(self):
        df = self.featureEncoder()
        return df
    
def Scale(df, scale=False,oversample=False):
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if oversample: 
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X,y)

    data = np.hstack((X, np.reshape(y,(-1, 1))))

    return data, X, y

def processNew(df):
    df['gender'] = df['gender'].map({'Male': 0,'Female': 1})
    df['diabetesMed'] = df['diabetesMed'].map({'No': 0,'Yes': 1})
    df['age'] = df['age'].map({'[0-10]': 1, '[10-20]': 2, '[20-30]': 3, '[30-40]': 4,'[40-50]': 5, '[50-60]': 6, '[60-70]': 7, '[70-80]': 8, '[80-90]': 9, '[90-100]': 10})
    return df
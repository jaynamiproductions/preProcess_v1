import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

class PreProcess:
    def __init__(self, file):
        self.df = pd.read_csv(file)

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
        return df[cond]
    
    def binaryMapper(self):
        df = self.noneDropper()
        df['gender'] = df['gender'].map({'Male': 0,'Female': 1})
        df['diabetesMed'] = df['diabetesMed'].map({'No': 0,'Yes': 1})
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 0, '<30': 1})
        df['age'] = df['age'].map({'[50-60]': 1, '[60-70]': 2, '[70-80]': 3, '[80-90]': 4, '[90-100]': 5})
        return df
    
    def featureEncoder(self):
        df = self.binaryMapper()
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(df[['race']]).toarray()

        vals = list(set(df['race'].values))

        for i in range(len(matrix.T)):
            df.insert(0, vals[i], matrix.T[i])

        return df.drop(['race'], axis=1)
    
    def processed(self):
        df = self.featureEncoder()
        return df.dropna()
    
def scale(df, oversample=False):
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample: 
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X,y)

    data = np.hstack((X, np.reshape(y,(-1, 1))))

    return data, X, y
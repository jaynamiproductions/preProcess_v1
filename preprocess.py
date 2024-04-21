import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class column_filter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cols = ['race', 'gender', 'age', 'time_in_hospital', 
                'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 
                'number_diagnoses', 'diabetesMed', 'readmitted']
        return X[cols]
    
class text_cleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['age'] = X['age'].str.replace(')',']')
        return X
    
class none_dropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[X['race'] != '?']
    
class binary_mapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['gender'] = X['gender'].map({'Male': 0,'Female': 1})
        X['diabetesMed'] = X['diabetesMed'].map({'No': 0,'Yes': 1})
        return X
    
class feature_encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['race']]).toarray()

        vals = list(set(X['race'].values)) + list(set(X['age'].values))

        for i in range(len(matrix.T)):
            X.insert(0, vals[i], matrix.T[i])

        return X.drop(['race', 'age'], axis=1)
    
def runPipeline(path):
    df = pd.read_csv(path)
    pipe = Pipeline([
        ('filter', column_filter()),
        ('cleaner', text_cleaner()),
        ('dropper', none_dropper()),
        ('mapper', binary_mapper()),
        ('encoder', feature_encoder())
    ])
    return pipe.fit_transform(df)
    
    
      
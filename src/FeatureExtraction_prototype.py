'''
Feature Extraction class

Authors:
    Xavier Paredes-Fortuny <xparedesfortuny@gmail.com>
    (add yourself if you add/modify anything)

Version: 1.0
Date Last Modification: 20/01/2016
'''

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler


class FeatureExtraction(object):
    """
    Data processing from pandas data frame
    """
    def __init__(self, num_col, cat_col, col_types):
        self.df = None
        self.X_cat = None
        self.X_num = None
        self.num_col = num_col
        self.cat_col = cat_col
        self.col_types = col_types
        self.h = FeatureHasher(n_features=10,
                               input_type='string',
                               non_negative=True)
        self.s = StandardScaler()
        self.init_standard_scaler()

    def init_standard_scaler(self):
        reader = pd.read_csv('test.csv', chunksize=1, usecols=self.num_col,
                             dtype=self.col_types)
        for row in reader:
            print row.as_matrix()
            self.s.partial_fit(row.as_matrix())

    def data_cleaning(self):
        self.df['gender'].replace('N', 'M', inplace=True)

    def get_features(self, df):
        """
        :param df: pandas data frame
        :return: x and y numpy arrays
        """
        y = df['click'].as_matrix()
        self.df = df.drop('click', 1)
        self.data_cleaning()
        self.X_num = self.s.transform(self.df[self.num_col].as_matrix())
        self.X_cat = self.h.transform(np.asarray(
                self.df[self.cat_col].astype(str))).toarray()
        return np.concatenate((self.X_num, self.X_cat), axis=1), y


def main():
    df = pd.DataFrame([['M', 'red', 30, 100, 1],
                       ['M', 'yellow', 35, 150, 0],
                       ['F', 'green', 30, 100, 0],
                       ['N', 'green', 50, 180, 1],
                       ['M', 'red', 50, 180, 1],
                       ['F', 'blue', 72, 150, 0]],
                      columns=['gender', 'color', 'age', 'height', 'click'])
    df.to_csv('test.csv', index=False)

    #=================================================================#

    col_types = {'age': float, 'height': float,
                 'gender': str,'color': str,
                 'click': int}
    num_col = ['age', 'height']
    cat_col = ['gender', 'color']
    cols_to_read = num_col+cat_col+['click']

    reader = pd.read_csv('test.csv', chunksize=1, usecols=cols_to_read,
                         dtype=col_types)
    f = FeatureExtraction(num_col, cat_col, col_types)
    for row in reader:
        x, y = f.get_features(row)
        print x, y


if __name__ == '__main__':
    main()

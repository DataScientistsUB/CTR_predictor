from __future__ import division
from time import time
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def read_data(fname, chunksize, cols_to_read):
    """Return a TextFileReader iterator"""
    additional_na_values = ['NoRating', 'Sin clasificar', '  ', ' ', '.']
    return pd.read_csv(fname, sep=';', header=0, dtype='str',
                       chunksize=chunksize, na_values=additional_na_values,
                       usecols=cols_to_read)


def data_cleaning(df, training=True):
    """Perform some data cleaning"""
    if training:
        df['booking'].fillna(0, inplace=True)
        df['booking'].ix[df['booking']=='True'] = 1
        df['booking'].ix[df['booking']=='False'] = 0
        df['booking'] = df['booking'].astype(int)

    to_replace = ['.0',
                  '* ', ' *',
                  '** ', ' **',
                  '*** ', ' ***',
                  '**** ', ' ****',
                  '***** ', ' *****',
                  '1\xc2\xaa Cat', '1\xc2\xaaCat', '1\xc2\xaa cat', '1\xc2\xaacat',
                  '2\xc2\xaa Cat', '2\xc2\xaaCat', '2\xc2\xaa cat', '2\xc2\xaacat']
    value = ['0',
             '*', '*',
             '**', '**',
             '***', '***',
             '****', '****',
             '*****', '*****',
             '1a_cat', '1a_cat', '1a_cat', '1a_cat',
             '2a_cat', '2a_cat', '2a_cat', '2a_cat']
    df['hotel_category'] = df['hotel_category'].replace(to_replace, value)
    return df


def get_bins(df, bin_list):
    """Return the data frame with the columns binned according to bin list"""
    for key, value in bin_list.iteritems():
        df[key] = pd.cut(df[key], value, include_lowest=True)
    return df


def get_weekday(df):
    """Get weekday for check in, check out, and ts"""
    df['check_in_weekday'] = pd.to_datetime(df['check_in']).dt.dayofweek
    df['check_out_weekday'] = pd.to_datetime(df['check_out']).dt.dayofweek
    df['ts_weekday'] = pd.to_datetime(df['ts']).dt.dayofweek
    return df.drop(['check_in', 'check_out', 'ts'], axis=1)


def get_categories(n, columns_to_encode):
    """Get all different categories from the first 'n' entries"""
    df = read_data('train.csv', n, columns_to_encode).get_chunk()
    print '\nGetting categories using {} entries from {} features...'.format(n, df.shape[1])
    print 'Number of possible values of "hotel category" feature: {}'.format(len(set(df['hotel_category'])))
    df = data_cleaning(df, False)
    print df.head()
    print 'Number of possible values of "hotel category" feature: {} (after data cleaning)'.format(len(set(df['hotel_category'])))
    col_list0 = dict()
    for col in columns_to_encode:
        col_list0[str(col)] = list(pd.get_dummies(df[col], dummy_na=False, columns=col, sparse=True))
    return col_list0


def set_manual_categories(col_list0):
    """Return the categories from col_list0"""
    adults_bins = range(0, 20, 1)+range(20, 200, 25)+[10000]
    children_bins = range(0, 20, 1)+range(20, 200, 25)+[10000]
    stay_bins = range(0, 30, 1)+range(30, 60, 2)+range(60, 100, 10)+[10000]
    forced_price_bins = [0, 1]
    check_in_weekday_bins = range(7)
    adv_bins = range(0, 30, 1)+range(30, 60, 2)+range(60, 100, 5)+range(100, 400, 10)+[365*10]
    average_bins = np.arange(0,10.1,0.1)
    price_bins = [0]+range(10, 200, 1)+range(200, 600, 5)+range(600, 1000, 10)+range(1000, 2000, 50)+[100000]
    bin_list = {'adv': adv_bins, 'average': average_bins, 'price': price_bins,
                'net': price_bins, 'adults': adults_bins, 'children': children_bins,
                'stay': stay_bins, 'check_in_weekday': check_in_weekday_bins,
                'check_out_weekday': check_in_weekday_bins,
                'ts_weekday': check_in_weekday_bins,
                'forced_price': forced_price_bins}
    print 'Setting the manual categories for {} features'.format(len(bin_list))
    for key, value in bin_list.iteritems():
        col_list0[str(key)] = [s for s in set(sorted(pd.cut(value, value, include_lowest=True)))]
    return col_list0, bin_list


def set_dict_vectorizer(col_list0):
    """Initialize the DictVectorizer() with the col_list0 feature dictionary"""
    print 'TOTAL number of features to be one hot encoded: {}'.format(len(col_list0))
    # Creates an homogeneous structured dictionary to feed the DictVectorizer
    max_length = 0
    for (k, v) in col_list0.iteritems():
        if max_length < len(col_list0[str(k)]):
            max_length = len(col_list0[str(k)])

    for key, value in col_list0.iteritems():
        col_list0[str(key)] += [col_list0[str(key)][0]]*(max_length-len(col_list0[str(key)]))

    # Initialize the DictVectorizer()
    vec = DictVectorizer()
    vec.fit(pd.DataFrame(col_list0).T.to_dict().values())
    print 'Number of one hot encoded features: {}'.format(len(vec.vocabulary_))
    return vec


def hot_encode_df(df, vec):
    """Perform one hot encoding"""
    return vec.transform(df.drop('booking', axis=1).T.to_dict().values()),\
           df['booking'].as_matrix()


def process_data_frame(df, encoded0, vec, columns_to_encode, bin_list):
    """Call data_cleaning(), get_weekeday(), get_bins(),
       and hot_encode_df() functions, and return X and y matrices"""
    df = data_cleaning(df)
    df = get_weekday(df)
    df = get_bins(df, bin_list)
    X, y = hot_encode_df(df, vec)
    return X, y


def model_training(X_train, y_train, model):
    """Perform partial fitting"""
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    return model


def compute_revenue(c):
    """Compute 'revenue' from the confusion matrix
       * Assumption: the ad is shown when a positive prediction of click and buy
         is made (FP+TP), hence if you get 1 you pay
       * Confusion matrix (sklearn):
       c = [TN FP]
           [FN TP]
         = T
           R
           U
           E
             PREDICTED
       * Revenue formula:
       revenue = times_clicked_and_buyed * money_per_sell - times_clicked * cost_per_click
               = TP * money_per_sell - (TP+FP) * cost_per_click
    """
    money_per_sell = 5.
    CPC = 0.5
    return c[1,1]*money_per_sell-(c[0,1]+c[1,1])*CPC


def evaluate_model(X_test, y_test, model):
    """Evaluate the model"""
    y_pred = model.predict(X_test)
    c = confusion_matrix(y_test, y_pred)
    print '\nModel score: {}'.format(model.score(X_test,y_test))
    print 'Classification report:\n{}'.format(classification_report(y_test, y_pred))
    print 'Confusion matrix:\n{}'.format(c)
    print 'Revenue: {} (the higher the better)'.format(compute_revenue(c))


def main():
    """
    Computes the booking prediction of an online ad

    The current script is a prototype aimed to provide a baseline model for
    the FTRL proximal implementation. It has been designed to work with large
    data sets thanks to pandas read_csv iterator and the partial_fit() capability
    of sklearn SGDClassifier (incremental learning / online learning). Aimed to
    mimic FTRL proximal implementations of CTR modeling (with millions of features),
    it performs One Hot Encoding for all the variables, including the numerical
    (non categorical) features. It also computes the weekday variables in
    substitution of all the date features, and conducts some data cleaning
    (mainly for correcting typos in hotel_category feature)

    Input:
    * train.csv (80 percent of raw.csv)
    * test.csv (last 20 percent of raw.csv)

    Set-up:
    * 'number_of_entries_to_get_possible_categories': is the number of entries
      used to determine all the possible categories of 'cols_to_auto_encode' features
    * 'chunksize': number of entries read by pandas at each partial fit iteration.
    """

    # SET-UP
    number_of_entries_to_get_possible_categories = 1000000
    chunksize = 100000
    #

    cols_to_auto_encode = ['hotel_category', 'city', 'hotel', 'provider']
    cols_to_read = cols_to_auto_encode+['adults', 'adv', 'average', 'children', 'net',
                                        'price', 'stay',
                                        'forced_price',
                                        'check_in', 'check_out', 'ts', 'booking']


    # Get all the different categories available in the first
    # 'number_of_entries_to_get_possible_categories' from 'cols_to_auto_encode' columns
    # plus it manually sets the categories in bin_list
    col_list0 = get_categories(number_of_entries_to_get_possible_categories, cols_to_auto_encode)
    col_list0, bin_list = set_manual_categories(col_list0)
    vec = set_dict_vectorizer(col_list0)


    # Initialize the SGDClassifier model
    # model = SGDClassifier(loss='log', n_iter=200, alpha=.0000001, penalty='l2',\
    #                       learning_rate='invscaling', power_t=0.5, eta0=4.0,
    #                       shuffle=True, n_jobs=-1, random_state=0,
    #                       class_weight={1:2})
    model = SGDClassifier(loss='hinge', class_weight={1:2}, penalty='elasticnet',
                          n_iter=100, shuffle=True)

    # Data processing and partial_fit modeling
    print '\nTraining...'
    reader = read_data('train.csv', chunksize, cols_to_read)
    t0 = time()
    for i, df in enumerate(reader):
        X_train, y_train = process_data_frame(df, col_list0, vec, cols_to_auto_encode, bin_list)
        model = model_training(X_train, y_train, model)
        if i // chunksize % 100000 == 0:
            print 'Chunk {} trained with shape {} ({:.2f} minutes)'.format(i, X_train.shape, (time()-t0)/60.)
            t0 = time()


    # Model evaluation
    print '\nEvaluating...'
    df = read_data('test.csv', None, cols_to_read)
    X_test, y_test = process_data_frame(df, col_list0, vec, cols_to_auto_encode, bin_list)
    evaluate_model(X_test, y_test, model)


if __name__ == '__main__':
    t0 = time()
    print '\ Booking prediction of an online ad /'
    main()
    print '\nNormal termination'
    print 'Elapsed time {:.2f} minutes'.format((time()-t0)/60.)

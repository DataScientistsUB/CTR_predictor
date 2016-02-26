'''
Computes the Click-Through Rate Prediction on accommodation Ads

Authors:
    Xavier Paredes-Fortuny <xparedesfortuny@gmail.com>
    and Luis Blanco <lblanco@gmail.com>
    (add yourself if you add/modify anything)

Attributions:
    The following script has been adapted from 'fast_solution_v3.py' which was
    written by Sam Hocevar for a Kaggle contest

Version: 1.0
Creation Date: 10/02/2016
Date Last Modification: 26/02/2016
Revision History:
    Rev 2.0 - Addition of the weighted log-loss, Luis Blanco
    Rev 1.1 - Addition of the confusion matrix, and manual tunning of L1 and L2
    Rev 1.0 - Addition of FTRL- proximal method and online CV, Luis Blanco
    Rev 0.1 - File Created, Xavier Paredes-Fortuny, 09/02/2016
'''



from IPython import get_ipython  
get_ipython().magic('reset -sf')


from csv import DictReader
from datetime import datetime
from math import exp, log, sqrt, floor
import scipy.io as sio 


##############################################################################
# Definition of the parameters
##############################################################################

# A, paths
#path = 'train\\train.csv'               # path to training file
train= 'raw.csv'
test = 'test\\test_rev2'                 # path to testing file
submission = 'submission1234.csv'  # path of to be outputted submission file

# B, model
alpha = 1.#.1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.    # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# LUIS: NEW PARAMETERS TO ADJUST THE LOGLOSS

gamma = 5.#0.92#0.01#0.92
delta = 0.005#0.08#0.99#0.08

# C, feature/hash trick
D = 2 ** 20#2 ** 20             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = None #9   # data after date N (exclusive) are used as validation
holdout = None#9 #None  # use every N training instance for holdout validation
hold_train = floor(681314*0.9)



############################################################################
##
############################################################################


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''
    col_names = ['id', 'adults', 'adv', 'average', 'booking', 'check_in',
                 'check_out', 'children', 'city', 'forced_price', 'hotel',
                 'hotel_category', 'net', 'price', 'provider', 'stay', 'ts']

    with open(path, 'r') as f:
        f.next()  # skip header
        for t, row in enumerate(DictReader(f, fieldnames=col_names, delimiter=';')):

            # process id
            ID = row['id']
            del row['id']

            # process booking
            y = 1
            if row['booking'] == '':
                y = 0
            assert row['booking'] in ['', 'True'], 'Booking value not considered'
            del row['booking']

            # data cleaning
            if row['hotel_category'] in ['* ', ' *']:
                row['hotel_category'] = '*'
            elif row['hotel_category'] in ['** ', ' **']:
                row['hotel_category'] = '**'
            elif row['hotel_category'] in ['*** ', ' ***']:
                row['hotel_category'] = '***'
            elif row['hotel_category'] in ['**** ', ' ****']:
                row['hotel_category'] = '****'
            elif row['hotel_category'] in ['***** ', ' *****']:
                row['hotel_category'] = '*****'
            elif row['hotel_category'] in ['1\xc2\xaa Cat', '1\xc2\xaaCat',
                                           '1\xc2\xaa cat', '1\xc2\xaacat']:
                row['hotel_category'] = '1a_cat'
            elif row['hotel_category'] in ['2\xc2\xaa Cat', '2\xc2\xaaCat',
                                           '2\xc2\xaa cat', '2\xc2\xaacat']:
                row['hotel_category'] = '2a_cat'

            # extract date (used in online learning)
            date = '{}{}{}{}{}'.format(row['ts'][:4], row['ts'][5:7],
                                       row['ts'][8:10], row['ts'][11:13],
                                       row['ts'][14:16])
            row['date_in'] = row['check_in'][:10]
            row['date_out'] = row['check_out'][:10]
            row['date_ts'] = row['ts'][:10]

            # extract time HH:MM (forget seconds)
            row['time_in'] = row['check_in'][11:16]
            row['time_out'] = row['check_out'][11:16]
            row['time_ts'] = row['ts'][11:16]

            # extract weekday
            row['weekday_in'] = str(datetime.strptime(row['date_in'], '%Y-%m-%d').weekday())
            row['weekday_out'] = str(datetime.strptime(row['date_out'], '%Y-%m-%d').weekday())
            row['weekday_ts'] = str(datetime.strptime(row['date_ts'], '%Y-%m-%d').weekday())

            del row['check_in']
            del row['check_out']
            del row['ts']

            # round float to integers (to avoid one new column for every decimal value)
            row['net'] = str(int(round(float(row['net']))))
            row['price'] = str(int(round(float(row['price']))))

            # build x
            x = []
            for key in row:
                value = row[key]

                # one-hot encode everything with hash trick
                index = abs(hash(key + '_' + value)) % D
                x.append(index)

            yield t, date, ID, x, y


#############################################################################
# Logloss
############################################################################

def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

##############################################################################
# FTRL-proximal algortihm: class and generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {} # LUIS: w is initialized as an empty dictionary

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i] # LUIS: wTx is the inner product w'*x

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.))) # LUIS: This may be an error, because wTx should be wTx = w'*x

    def update(self, x, p, y,gamma,delta):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = delta*p - gamma*y + (gamma-delta)*y*p
        #g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


###########################################################################
#    START TRAINING
##########################################################################3

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# start training
p_test = []
y_test = []
for e in xrange(epoch):
    loss = 0.
    count = 0
    count_2 = 0

    for t, date, ID, x, y in data(train, D):  # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)



        # step 1, get prediction from learner
        p = learner.predict(x)

        #if (holdafter and date > holdafter) or (holdout and t % holdout == 0):
        if (hold_train and t > hold_train)or (holdafter and date > holdafter) or (holdout and t % holdout == 0):
            # step 2-1, calculate validation loss
            #           we do not train with the validation data so that our
            #           validation loss is an accurate estimation
            #
            # holdafter: train instances from day 1 to day N
            #            validate with instances from day N + 1 and after
            #
            # holdout: validate with every N instance, train with others
            
            if count ==0:
                print('Validation set') 
                
            loss += logloss(p, y)
            count += 1
            #print count
            p_test.append(p)
            y_test.append(y)
        else:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y,gamma, delta)
            #count_2 += 1
            
            #if (count_2 % 1000 == 0):
            if (t% 1000 == 0):
                #print count_2
                print t

    # from sklearn.metrics import confusion_matrix
    # for i, p in enumerate(p_test):
    #     if p < 0.5:
    #         p_test[i] = 0
    #     else:
    #         p_test[i] = 1
    # print confusion_matrix(y_test, p_test)
    print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (e, loss/count, str(datetime.now() - start)))

with open('y_y_pred.txt', 'wb') as f:
    for i, _ in enumerate(y_test):
        f.write(str(y_test[i])+' '+str(p_test[i])+'\n')
        






###########################################################################
#
###########################################################################

#def main():
#    D = 2 ** 20  # number of weights to use
#    path = 'train.csv'
#
#    # for t, date, ID, x, y in data(path, D):
#        # online cross-validation
#        # update learner
#
#
#if __name__ == '__main__':
#    main()

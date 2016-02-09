'''
Computes the Click-Through Rate Prediction on accommodation Ads

Authors:
    Xavier Paredes-Fortuny <xparedesfortuny@gmail.com>
    (add yourself if you add/modify anything)

Attributions:
    The following script has been adapted from 'fast_solution_v3.py' which was
    written by Sam Hocevar for a Kaggle contest

Version: 0.1
Creation Date: 09/02/2016
Date Last Modification: 09/02/2016
Revision History:
    Rev 0.1 - File Created, Xavier Paredes-Fortuny, 09/02/2016
'''


from csv import DictReader
from datetime import datetime


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


def main():
    D = 2 ** 20  # number of weights to use
    path = 'train.csv'

    # for t, date, ID, x, y in data(path, D):
        # online cross-validation
        # update learner


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
from os.path import exists

path = 'wisdm-dataset/raw'
#   Load the dataset
columns = ['user', 'activity', 'timestamp', 'x-acc', 'y-acc', 'z-acc']
inv_dict = {'A': 'walking',
            'B': 'jogging',
            'C': 'stairs',
            'D': 'sitting',
            'E': 'standing',
            'F': 'typing',
            'G': 'teeth',
            'H': 'soup',
            'I': 'chips',
            'J': 'pasta',
            'K': 'drinking',
            'L': 'sandwich',
            'M': 'kicking',
            'O': 'catch',
            'P': 'dribbling',
            'Q': 'writing',
            'R': 'clapping',
            'S': 'folding'}
act_dict = {'walking': 'A', 'jogging': 'B', 'stairs': 'C', 'sitting': 'D', 'standing': 'E', 'typing': 'F',
            'teeth': 'G', 'soup': 'H', 'chips': 'I', 'pasta': 'J', 'drinking': 'K', 'sandwich': 'L',
            'kicking': 'M', 'catch': 'O', 'dribbling': 'P', 'writing': 'Q', 'clapping': 'R', 'folding': 'S'}


def import_all_data():
    max_len = 4000
    for device in ['watch', 'phone']:
        for user in range(51):
            user_data = pd.read_csv(path + '/' + device + '/gyro/data_' + str(user + 1600) + '_gyro_'+device+'.txt', header=None, names=columns,
                                    sep=',', lineterminator=';')
            for activity in act_dict:
                temp_df = user_data[user_data['activity'] == act_dict[activity]]
                temp_df = temp_df[['x-acc', 'y-acc', 'z-acc']].values
                temp_df = temp_df[:max_len]
                if len(temp_df) > 10:
                    if len(temp_df) < max_len:
                        max_len = len(temp_df)
                    if "df" not in locals():
                        df = temp_df.astype(np.float)
                    elif df.ndim == 2:
                        df = np.stack((df, temp_df.astype(np.float)), axis=2)
                    else:
                        df = np.concatenate((df[:max_len, :, :], temp_df[:, :, np.newaxis].astype(np.float)), axis=2)
    return df


def import_oscillatory_data():
    max_len = 3600
    watch_activities = ['walking', 'jogging', 'dribbling', 'clapping']
    phone_activities = ['walking', 'jogging']
    for user in range(51):
        user_data = pd.read_csv(path + '/watch/gyro/data_' + str(user + 1600) + '_gyro_watch.txt', header=None, names=columns,
                                sep=',', lineterminator=';')
        for activity in watch_activities:
            temp_df = user_data[user_data['activity'] == act_dict[activity]]
            temp_df = temp_df[['x-acc', 'y-acc', 'z-acc']].values
            temp_df = temp_df[:max_len]
            if len(temp_df) > 10:
                if len(temp_df) < max_len:
                    max_len = len(temp_df)
                if "df" not in locals():
                    df = temp_df.astype(np.double)
                elif df.ndim == 2:
                    df = np.stack((df, temp_df.astype(np.double)), axis=2)
                else:
                    df = np.concatenate((df[:max_len, :, :], temp_df[:, :, np.newaxis].astype(np.double)), axis=2)

    for user in range(51):
        user_data = pd.read_csv(path + '/phone/gyro/data_' + str(user + 1600) + '_gyro_phone.txt', header=None, names=columns,
                                sep=',', lineterminator=';')
        for activity in phone_activities:
            temp_df = user_data[user_data['activity'] == act_dict[activity]]
            temp_df = temp_df[['x-acc', 'y-acc', 'z-acc']].values
            temp_df = temp_df[:max_len]
            if len(temp_df) > 10:
                if len(temp_df) < max_len:
                    max_len = len(temp_df)
                else:
                    df = np.concatenate((df[:max_len, :, :], temp_df[:, :, np.newaxis].astype(np.double)), axis=2)
    return df


def load_all_data():
    load_data = 0
    if exists('wisdm_data.npy'):
        with open('wisdm_data.npy', 'rb') as data:
            all_data = np.load(data)
    else:
        print('creating data file')
        all_data = import_all_data()
        with open('wisdm_data.npy', 'wb') as data:
            np.save(data, all_data)
    return all_data


def load_oscillatory_data():

    if exists('wisdm_oscillatory.npy'):
        with open('wisdm_oscillatory.npy', 'rb') as data:
            oscillatory_data = np.load(data)
    else:
        print('creating data file')
        oscillatory_data = import_oscillatory_data()
        with open('wisdm_oscillatory.npy', 'wb') as data:
            np.save(data, oscillatory_data)
    return oscillatory_data

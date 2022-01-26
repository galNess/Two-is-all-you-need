import numpy as np


def order_data(data, sample_len, val_split=.1):
    data /= 5  # Rough normalization of data
    data = data[sample_len//2:-sample_len//2, :, np.random.permutation(np.shape(data)[2])]  # Chopping edges
    chunk_num = int(np.shape(data)[0]*val_split/sample_len)                       # N chunks per long sample
    leftover_times = int(np.shape(data)[0] % chunk_num)                    # Time buffer for random cropping
    data = data[:-leftover_times, :, :].transpose(1, 2, 0)\
        .reshape(3, int(np.shape(data)[2]*chunk_num), -1).transpose(1, 2, 0)
    val_data = mean_std_norm(data[:, :sample_len, :])      # Validation data is already in sample_len length
    train_data = data[:, sample_len:, :]                 # Training data will be randomly chopped each epoch
    return train_data, val_data


def reshuffle_train_data(train_data, sample_len):
    # This function randomly chops the training data and reshuffles it
    start_idx = np.random.randint(np.shape(train_data)[1] % sample_len)   # The argument of randint is the leftover time
    train_data = train_data[:, start_idx:start_idx+(np.shape(train_data)[1]//sample_len)*sample_len, :]
    train_data = train_data.transpose(2, 0, 1).reshape(3, -1, sample_len).transpose(1, 2, 0)
    train_data = train_data[np.random.permutation(np.shape(train_data)[0]), :, :]
    train_data = mean_std_norm(train_data)
    return train_data


def get_batch(source, first_sample, batch_size, axis_order):
    samples_num = min(batch_size, np.shape(source)[0] - 1 - first_sample)
    data = source[first_sample:first_sample + samples_num, :, axis_order]    # axis_order is used for shuffling the axes
    input_xy = data.clone().unsqueeze(3).permute(0, 1, 3, 2)
    target_z = data[:, :, 2].unsqueeze(2)
    input_xy = input_xy[:, :, :, 0:2]
    return input_xy, target_z


def mean_std_norm(data):
    for d in range(np.shape(data)[2]):
        data[:, :, d] = data[:, :, d] - data[:, :, d].mean(axis=1)[:, np.newaxis]
        data[:, :, d] = data[:, :, d] / data[:, :, d].std(axis=1)[:, np.newaxis]
    return data



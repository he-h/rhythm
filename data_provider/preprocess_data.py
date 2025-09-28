import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import ceil, floor

    
def preprocess_yj_data(root_path, city='D', scale=False, size=(7*48, 6*48, 1*48), interval=1*48, flag='train'):
    """
    Preprocesses the YjMob100k dataset and saves it as an .npz file.
    input_seq_feature: ['User ID', 'time slot', 'DayOfWeek', 'Day', 'Latitude', 'Longitude', 'Place ID']
    predict_seq_feature: ['User ID', 'time slot', 'DayOfWeek', 'Day']
    predict_seq_label: ['Place ID']

    Args:
        root_path (str): Root directory path where the dataset is located.
        city (str): City name, ['A', 'B', 'C', 'D']. 100k, 25k, 20k, 6k, -3k
        scale (bool): Whether to scale the numerical features.
        output_filename (str): Name of the output .npz file.
    """
    dataset = load_yj_df(city=city)
    total_users = dataset['uid'].nunique()
    
    samples_in_a_day = 48
    input_seq_length, _, predict_seq_length = size
    input_seq_length, predict_seq_length = input_seq_length // samples_in_a_day, predict_seq_length // samples_in_a_day
    print(f"Input sequence length: {input_seq_length} days, Prediction sequence length: {predict_seq_length} days")
    
    # flag = 'train' 'val' 'test', split the dataset 7:2:1 based on days (dataset['d'])
    total_days = dataset['d'].max() + 1
    if flag == 'train':
        start_day, end_day = 0, total_days * 0.7
    elif flag == 'val':
        start_day, end_day = total_days * 0.7 - input_seq_length, total_days * 0.9
    elif flag == 'test':
        start_day, end_day = total_days * 0.9 - input_seq_length, total_days
        
    if city == 'BOS':
        if flag == 'train':
            start_day, end_day = 0, ceil(total_days * 0.6)
        elif flag == 'test':
            start_day, end_day = total_days - 14, total_days
        elif flag == 'val':
            start_day, end_day = ceil(total_days * 0.6) - 7, total_days - 7
    
    start_day = ceil(start_day)
    dataset = dataset[dataset['d'].between(start_day, end_day)]
    print(f"Total users: {total_users}, Total days: {total_days}, Total records: {len(dataset)}")
    print(f"Splitting data for {flag} set: {start_day} to {end_day}")

    day_values = np.arange(start_day, end_day)

    # Process sequences
    input_seq_feature = []
    predict_seq_feature = []
    predict_seq_label = []

    # Group data by user
    grouped_data = dataset.groupby('uid')
    print(f"Number of unique uid: {len(grouped_data)}")
    set_uids = np.arange(total_users)
    print(f"Missing uid is {set_uids[~np.in1d(set_uids, dataset['uid'].unique())]}") # 1785 4204

    for uid, uid_df in tqdm(grouped_data, desc="Generating sequences"):
        full_seq_x = generate_yj_sequence(uid_df, day_values)
        # Calculate the number of valid sequences
        num_seq = len(day_values) - input_seq_length - predict_seq_length + 1
        if num_seq > 0:
            # Precompute indices for slicing
            input_start_indices = np.arange(num_seq) * samples_in_a_day
            input_end_indices = input_start_indices + input_seq_length * samples_in_a_day
            predict_start_indices = input_end_indices
            predict_end_indices = predict_start_indices + predict_seq_length * samples_in_a_day

            # Slice sequences in bulk
            input_seq_feature.extend([full_seq_x[start:end] for start, end in zip(input_start_indices, input_end_indices)])
            predict_seq_feature.extend([full_seq_x[start:end, 0:4] for start, end in zip(predict_start_indices, predict_end_indices)])
            predict_seq_label.extend([full_seq_x[start:end, -1] for start, end in zip(predict_start_indices, predict_end_indices)])

    # Convert lists to numpy arrays and then to torch tensors
    print(f"Total sequences generated: {len(input_seq_feature)}")
    
    input_seq_feature = np.array(input_seq_feature, dtype=np.float32)
    predict_seq_feature = np.array(predict_seq_feature, dtype=np.float32)
    predict_seq_label = np.array(predict_seq_label, dtype=np.int64)
    print(f"Input sequence shape: {input_seq_feature.shape}, Prediction sequence shape: {predict_seq_feature.shape}, Prediction label shape: {predict_seq_label.shape}")

    output_filename = f"yj_{city}_size_{size[0]}_{size[1]}_{size[2]}_{flag}.npz"
    np.savez_compressed(
        os.path.join(root_path, f"dataset/yj/{output_filename}"),
        input_seq_feature=input_seq_feature,
        predict_seq_feature=predict_seq_feature,
        predict_seq_label=predict_seq_label,
    )
    print(f"Preprocessed data saved to {output_filename}")
    return

def generate_yj_sequence(data_by_day, days):
    """
    Generate sequences for each user.

    Args:
        data_by_day (pd.DataFrame): Data for a specific user.
        days (list): List of unique days.

    Returns:
        np.ndarray: Full sequence features for the user.
    """
    uid = data_by_day['uid'].iloc[0]  # User ID
    time_steps = np.arange(48)  # 48 time slots in a day
    seq_x = []

    # Iterate over each day
    for d in days:
        # Initialize default values for the entire day
        full_day_x = np.full((48, 7), [-1, -1, -1, -1, 999, 999, 40000], dtype=np.float32)
        full_day_x[:, 0] = uid  # Set User ID
        full_day_x[:, 1] = time_steps  # Set time slots
        full_day_x[:, 2] = d % 7  # Day of the week (0-6)
        full_day_x[:, 3] = d  # Day

        # Check if the day exists in the user's data
        if d in data_by_day['d'].values:
            day_data = data_by_day[data_by_day['d'] == d].set_index('t')  # Set time as index

            # Extract valid time slots
            matching_times = day_data.index.values
            full_day_x[matching_times, -3] = day_data['x'].values  # Latitude
            full_day_x[matching_times, -2] = day_data['y'].values  # Longitude
            full_day_x[matching_times, -1] = day_data['label'].values  # Place ID

        seq_x.append(full_day_x)  # Add the day's data to the sequence

    return np.concatenate(seq_x, axis=0)  # Concatenate all days into one array
        
def yj_full4prompt(root_path='', city='D'):
    dataset = load_yj_df(city=city)
    
    uids = dataset['uid'].unique()
    
    full_seq = []
    
    for uid in tqdm(uids):
        uid_full_seq = generate_yj_sequence(dataset[dataset['uid'] == uid], np.arange(0, 75))
        uid_full_seq = uid_full_seq.reshape(75, 48, 7)
        full_seq.append(uid_full_seq)
    
    full_seq = np.array(full_seq, dtype=np.float32)
    print(f"Full sequence shape: {full_seq.shape}")
    
    # save as npy
    output_filename = f"yj_{city}_full_sequence.npy"
    np.save(os.path.join(root_path, f"dataset/yj/{output_filename}"), full_seq)
    return
    

def normalize_uids(df, uid_col='uid'):
    """
    Normalizes the 'uid' column in a DataFrame by making the IDs consecutive
    while maintaining the relative order.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a column containing non-consecutive integer UIDs.
    uid_col (str): The column name containing the UID values.

    Returns:
    pd.DataFrame: DataFrame with UIDs normalized to consecutive integers.
    """
    # Get unique, sorted UIDs
    unique_uids = sorted(df[uid_col].unique())

    # Create a mapping from old UID to new consecutive UID
    uid_mapping = {old_uid: new_uid for new_uid, old_uid in enumerate(unique_uids)}

    # Apply the mapping to the DataFrame
    df[uid_col] = df[uid_col].map(uid_mapping)

    return df

def load_yj_df(city='D'):
    '''
    City B remove uid [1785 4204] with incomplete data
    '''
    if city in ['B', 'C', 'D']:
        data_path = os.path.join(root_path, f"dataset/yj/city{city}_challengedata.csv.gz")
    elif city == 'A':
        data_path = os.path.join(root_path, "dataset/yj/cityA_groundtruthdata.csv.gz")
    else:
        raise ValueError("Invalid city name. Choose from ['A', 'B', 'C', 'D']")
    
    print(f"Loading data from {data_path}")
    dataset = pd.read_csv(data_path, compression='gzip')
    
        
    dataset = normalize_uids(dataset) 
    
    if city in ['A', 'B', 'C', 'D']:
        total_users = dataset['uid'].nunique() - 3000 # Remove last 3000 users # TODO: change here
        print(f"Total users: {total_users}")
        dataset = dataset[dataset['uid'] < total_users]
    else:
        total_users = dataset['uid'].nunique()
        print(f"Total users: {total_users}")
    
    dataset['label'] = 200 * (dataset['x'] - 1) + (dataset['y'] - 1)
    dataset.sort_values(by=['uid', 'd', 't'], inplace=True)
    
    return dataset
    
    


if __name__ == "__main__":
    seq_len, label_len, pred_len = 7*48, 6*48, 1*48
    interval = 1*48
    size = (seq_len, label_len, pred_len)
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    city = 'B'
    yj_full4prompt(root_path, city=city)
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='train')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='val')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='test')
    
    city = 'C'
    yj_full4prompt(root_path, city=city)
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='train')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='val')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='test')
    
    city = 'D'
    yj_full4prompt(root_path, city=city)
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='train')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='val')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='test')

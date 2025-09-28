import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe, format_timedelta, convert_time_slot_to_str
import warnings
warnings.filterwarnings('ignore')



class Dataset_YJ(Dataset):
    def __init__(self, root_path, data_path=None, flag='train', size=None, llm_ckp_dir='meta-llama/Llama-3.2-1B',
                 scale=False, city=None, **kwargs):
        self.preprocessed_filename = f"yj_{city}_size_336_288_48_{flag}.npz"
        model_abbrev = llm_ckp_dir.split('/')[-1]
        # self.preprocessed_prompt_filename_x = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_1B_x.pt"
        # self.preprocessed_prompt_filename_y = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_{flag}_1B_y.pt"
        self.preprocessed_prompt_filename_x = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_{model_abbrev}_x.pt"
        self.preprocessed_prompt_filename_y = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_{flag}_{model_abbrev}_y.pt"
        self.seq_len = size[0] if size and len(size) > 0 else 40
        self.label_len = size[1] if size and len(size) > 1 else 30
        self.pred_len = size[2] if size and len(size) > 2 else 10
        self.token_len = self.seq_len - self.label_len  # 48
        self.token_num = self.seq_len // self.token_len  # 7
        self.flag = flag
        self.scale = scale

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        assert city in ['A', 'B', 'C', 'D', 'BOS']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"{self.preprocessed_filename}")
        self.embed_prompt_path_x = os.path.join(root_path, f"{self.preprocessed_prompt_filename_x}")
        self.embed_prompt_path_y = os.path.join(root_path, f"{self.preprocessed_prompt_filename_y}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        # self.__split_data__()  # Perform the split
        print(f"Loaded {self.num_samples} sequences from {self.flag} set.")

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']  # [num_samples, seq_len, 7]
        self.predict_seq_feature = data['predict_seq_feature']  # [num_samples, pred_len, 4]
        self.predict_seq_label = data['predict_seq_label']  # [num_samples, pred_len]
        self.num_samples = len(self.input_seq_feature)

        self.num_classes = 40001
        # print(self.embed_prompt_path_x)
        self.embed_prompts_x = torch.load(self.embed_prompt_path_x)
        self.embed_prompts_y = torch.load(self.embed_prompt_path_y)

    def get_num_class(self):
        return 40001 # 40000 for missing

    def get_num_users(self):
        return len(np.unique(self.input_seq_feature[:, 0, 0]))

    def __getitem__(self, index):
        """
        Retrieves the data sample at the specified index.
        ['User ID', 'time slot', 'DayOfWeek', 'Day', 'Latitude', 'Longitude', 'Place ID']
        Returns:
            tuple: (
                input_seq_feature (torch.Tensor),      # [seq_len, 7]
                predict_seq_feature (torch.Tensor),    # [pred_len, 4]
                predict_seq_label (torch.Tensor),      # [pred_len]
                prompt_embedding_x (torch.Tensor),     # [token_num, prompt_embedding_size]
                prompt_embedding_y (torch.Tensor]     # [prompt_embedding_size]
            )
        """
        input_feat = self.input_seq_feature[index]  # [seq_len, 7]
        output_feat = self.predict_seq_feature[index]  # [pred_len, 4]
        output_label = self.predict_seq_label[index]  # [pred_len]
        
        uid = int(input_feat[0, 0])
        start_day = int(input_feat[0, 3])
        indices = (uid * 75 + start_day + torch.arange(self.token_num)).long()
        embedded_token_prompt = self.embed_prompts_x[indices]
        embedded_token_prompt = embedded_token_prompt.view(self.token_num, -1)  # [token_num, prompt_embedding_size]

        return (
            torch.tensor(input_feat, dtype=torch.float32),  # [seq_len, 7]
            torch.tensor(output_feat, dtype=torch.float32),  # [pred_len, 4]
            torch.tensor(output_label, dtype=torch.int),  # [pred_len]
            embedded_token_prompt,  # [token_num, prompt_embedding_size] 
            self.embed_prompts_y[index],
        )

    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples


class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        
    # TODO: Prompt
    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        if self.data_set_type in ['traffic', 'electricity', 'ETTh1', 'ETTh2']:
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type == 'weather':
            end = (start + datetime.timedelta(minutes=10*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type in ['ETTm1', 'ETTm2']:
            end = (start + datetime.timedelta(minutes=15*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)


    
    

class Dataset_Preprocess_YJ_Token(Dataset):
    def __init__(self, root_path, size=None,
                 scale=False, city='D'):
        self.seq_len = size[0] if size and len(size) > 0 else 7*48
        self.label_len = size[1] if size and len(size) > 1 else 6*48
        self.pred_len = size[2] if size and len(size) > 2 else 48
        
        self.token_len = self.seq_len - self.label_len # 48
        self.token_num = self.seq_len // self.token_len # 7
        self.scale = scale
        preprocessed_filename = f"yj_{city}_full_sequence.npy"

        self.root_path = root_path
        
        assert city in ['A', 'B', 'C', 'D', 'BOS']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"dataset/yj/{preprocessed_filename}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        

    def __read_data__(self):
        """
        Loads data from the preprocessed .npy file.
        """
        self.data = np.load(self.preprocessed_path, allow_pickle=True)
        self.num_uids = self.data.shape[0]
        self.num_days = self.data.shape[1]
    

    def __getitem__(self, index):
        """
        Generate the prompt with enhanced temporal details, key transitions, and stay durations.
        """
        user_id = index // self.num_days
        day_id = index % self.num_days
        day_of_week_id = day_id % 7
        
        # Retrieve the data
        if self.city == 'BOS':
            day_of_week_dict = {0: 'Wednesday', 1: 'Thursday', 2: 'Friday', 3: 'Saturday', 4: 'Sunday', 5: 'Monday', 6: 'Tuesday'}
        else:
            day_of_week_dict = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        
        input_feat = self.data[user_id, day_id]  # [seq_len, 7]
        
        # Process trajectory data
        record_list = []
        user_id_val = int(input_feat[0, 0])
        
        # Track points for transition and stay analysis
        valid_points = []
        
        for j in range(0, self.token_len):
            latitude = input_feat[j, 4]
            if latitude == 999:
                continue
                
            longitude = input_feat[j, 5]
            time_slot = int(input_feat[j, 1])
            time_str = convert_time_slot_to_str(time_slot)
            
            # Add to record list
            record_list.append(f"{time_str}: (X={int(latitude)}, Y={int(longitude)})")
            
            # Store valid point
            valid_points.append({
                'coords': (int(latitude), int(longitude)),
                'time_slot': time_slot,
                'time_str': time_str
            })
        
        # Basic prompt without transitions or stays
        prompt = (
            f"This is the trajectory of user {user_id_val} of day {int(day_id)} which is a {day_of_week_dict[int(day_of_week_id)]}. "
            f"The trajectory consists of {len(record_list)} records, each record of coordinate is as follows: {'; '.join(record_list)}. "
        )
        
        # Only add transitions and stays if we have enough data points
        if len(valid_points) >= 3:
            # Define significant transition threshold (adjust based on coordinate scale)
            TRANSITION_THRESHOLD = 5
            
            # Identify transitions and stays
            transitions = []
            stay_locations = []
            current_cluster = [valid_points[0]]
            
            for i in range(1, len(valid_points)):
                prev_point = valid_points[i-1]
                curr_point = valid_points[i]
                
                # Calculate distance
                distance = ((curr_point['coords'][0] - prev_point['coords'][0])**2 + 
                            (curr_point['coords'][1] - prev_point['coords'][1])**2)**0.5
                
                if distance > TRANSITION_THRESHOLD:
                    # Found a transition
                    
                    # Process previous cluster as a stay
                    if len(current_cluster) >= 2:
                        avg_lat = sum(p['coords'][0] for p in current_cluster) / len(current_cluster)
                        avg_lon = sum(p['coords'][1] for p in current_cluster) / len(current_cluster)
                        duration = (current_cluster[-1]['time_slot'] - current_cluster[0]['time_slot']) / 2  # in hours
                        
                        if duration >= 0.5:  # Only record stays of at least 30 minutes
                            stay_locations.append({
                                'coords': (int(avg_lat), int(avg_lon)),
                                'start': current_cluster[0]['time_str'],
                                'end': current_cluster[-1]['time_str'],
                                'duration': duration
                            })
                    
                    # Record transition
                    transitions.append({
                        'from': prev_point['coords'],
                        'to': curr_point['coords'],
                        'time': curr_point['time_str'],
                        'distance': distance
                    })
                    
                    # Start new cluster
                    current_cluster = [curr_point]
                else:
                    # Continue current cluster
                    current_cluster.append(curr_point)
            
            # Process final cluster
            if len(current_cluster) >= 2:
                avg_lat = sum(p['coords'][0] for p in current_cluster) / len(current_cluster)
                avg_lon = sum(p['coords'][1] for p in current_cluster) / len(current_cluster)
                duration = (current_cluster[-1]['time_slot'] - current_cluster[0]['time_slot']) / 2
                
                if duration >= 0.5:
                    stay_locations.append({
                        'coords': (int(avg_lat), int(avg_lon)),
                        'start': current_cluster[0]['time_str'],
                        'end': current_cluster[-1]['time_str'],
                        'duration': duration
                    })
            
            # Add key transitions to prompt (if any)
            if transitions:
                # Sort by distance (largest first) and take top 3
                transitions.sort(key=lambda x: x['distance'], reverse=True)
                transition_strs = []
                
                for t in transitions[:min(3, len(transitions))]:
                    transition_strs.append(
                        f"At {t['time']}: (X={t['from'][0]}, Y={t['from'][1]}) → (X={t['to'][0]}, Y={t['to'][1]})"
                    )
                
                prompt += f"\n\nKey transitions: {'; '.join(transition_strs)}."
            
            # Add stay locations to prompt (if any)
            if stay_locations:
                # Sort by duration (longest first) and take top 3
                stay_locations.sort(key=lambda x: x['duration'], reverse=True)
                stay_strs = []
                
                for s in stay_locations[:min(3, len(stay_locations))]:
                    stay_strs.append(
                        f"(X={s['coords'][0]}, Y={s['coords'][1]}) from {s['start']} to {s['end']} ({s['duration']:.1f} hours)"
                    )
                
                prompt += f"\n\nMain stay locations: {'; '.join(stay_strs)}."
        
        return prompt


    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_uids * self.num_days
    
class Dataset_Preprocess_YJ(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 scale=False, city='D'):
        self.seq_len = size[0] if size and len(size) > 0 else 7*48
        self.label_len = size[1] if size and len(size) > 1 else 6*48
        self.pred_len = size[2] if size and len(size) > 2 else 48
        
        self.token_len = self.seq_len - self.label_len # 48
        self.token_num = self.seq_len // self.token_len # 7
        self.flag = flag
        self.scale = scale
        preprocessed_filename = f"yj_{city}_size_{self.seq_len}_{self.label_len}_{self.pred_len}_{self.flag}.npz"

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        
        assert city in ['A', 'B', 'C', 'D', 'BOS']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"dataset/yj/{preprocessed_filename}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']             # [num_samples, seq_len, 7]
        self.predict_seq_feature = data['predict_seq_feature']         # [num_samples, pred_len, 4]
        self.predict_seq_label = data['predict_seq_label']             # [num_samples, pred_len]
        self.num_samples = len(self.input_seq_feature)
        print(f"Loaded {self.num_samples} sequences from preprocessed data.")
    

    def __getitem__(self, index):
        """
        Generate an enhanced prompt that clearly defines the mobility prediction task.
        """
        # Retrieve the data
        if self.city == 'BOS':
            day_of_week_dict = {0: 'Wednesday', 1: 'Thursday', 2: 'Friday', 3: 'Saturday', 4: 'Sunday', 5: 'Monday', 6: 'Tuesday'}
        else:
            day_of_week_dict = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        
        input_feat = self.input_seq_feature[index]  # [seq_len, 7]
        user_id = int(input_feat[0, 0])
        day = int(self.predict_seq_feature[index][0, 3])
        day_of_week = day_of_week_dict[input_feat[0, 2]]
        
        prompt = (
            "You are a mobility prediction assistant that forecasts human movement patterns in urban environments. "
            "The city is represented as a 200 x 200 grid of cells, where each cell is identified by coordinates (X,Y). "
            "The X coordinate increases from left (0) to right (199), and the Y coordinate increases from top (0) to bottom (199). "
            "\n\n"
            f"TASK: Based on User {user_id}'s historical movement patterns, predict their locations for Day {day} ({day_of_week}). "
            "The predictions should capture expected locations at 30-minute intervals throughout the day (48 time slots). "
            "The model should analyze patterns like frequent locations, typical daily routines, and time-dependent behaviors "
            "to generate accurate predictions of where this user is likely to be throughout the next day."
            "\n\n"
            "The previous days' trajectory data contains information about the user's typical movement patterns, regular visited locations, "
            "transition times, and duration of stays. Key patterns to consider include: home and work locations, morning and evening routines, "
            "lunch-time behaviors, weekend vs. weekday differences, and recurring visit patterns."
        )
        
        return prompt


    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples
    
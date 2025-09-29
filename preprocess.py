import argparse
import torch
import os
from tqdm import tqdm
from models.preprocess import Model

from data_provider.data_loader import Dataset_Preprocess_Foursquare, Dataset_Preprocess, Dataset_Preprocess_YJ,\
    Dataset_Preprocess_YJ_Token, Dataset_Preprocess_US_Token, Dataset_Preprocess_US
from data_provider.preprocess_data import preprocess_foursquare_data, preprocess_yj_data
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default='meta-llama/Llama-3.1-8B', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='yj', 
                        help='dataset to preprocess')
    parser.add_argument('--city', type=str, default='D', help='city name')
    args = parser.parse_args()
    print(args)
    
    flags = ['train', 'test', 'val']
    model = Model(args)

    seq_len = 7*48
    label_len = 6*48
    pred_len = 1*48
    
    print(f'Preprocessing {args.dataset} dataset')
    
    if args.dataset == 'yj':
        root_path = ''
        
        data_sets = []
        for flag in flags:
            data_set = Dataset_Preprocess_YJ(
                root_path='',
                size=[seq_len, label_len, pred_len],
                city=args.city,
                flag=flag)
            data_sets.append(data_set)
            
        data_set_token = Dataset_Preprocess_YJ_Token(
            root_path='',
            size=[seq_len, label_len, pred_len],
            city=args.city)
    
        
    batch_size = 128     
    
    data_loader_token = DataLoader(
        data_set_token,
        batch_size=batch_size,
        shuffle=False,
    )
    model_abbr = args.llm_ckp_dir.split('/')[-1]
    
    save_dir_path = './dataset/'
    
    x_list = []
    for idx, data0 in tqdm(enumerate(data_loader_token), total=len(data_loader_token)):
        output = model(data0)
        x_list.append(output.detach().cpu())
    x_result = torch.cat(x_list, dim=0)
    print(f"x_result shape: {x_result.shape}")
    torch.save(x_result, save_dir_path + f'/{args.dataset}/{args.dataset}_{args.city}_{seq_len}_{label_len}_{pred_len}_{model_abbr}_x.pt')
    
    # y prompt
    
    for i, flag in enumerate(flags):
        data_loader = DataLoader(
            data_sets[i],
            batch_size=batch_size,
            shuffle=False,
        )
        y_list= []
        for idx, data1 in tqdm(enumerate(data_loader), total=len(data_loader)):
            output = model(data1)
            y_list.append(output.detach().cpu())
        
        y_result = torch.cat(y_list, dim=0)
        print(f"y_result shape: {y_result.shape}")
        torch.save(y_result, save_dir_path + f'/{args.dataset}/{args.dataset}_{args.city}_{seq_len}_{label_len}_{pred_len}_{flag}_{model_abbr}_y.pt')
    
    print(f'{args.dataset} dataset city {args.city} preprocessing finished')

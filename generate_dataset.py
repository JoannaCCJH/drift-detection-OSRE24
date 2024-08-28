import yaml
import argparse
import os
import pandas as pd
import numpy as np

from drifts.utils import preprocess, save_plots, postprocess
from drifts.predict import moving_prediction

def main(args):
    
    # Create output directory
    output_path = f'{args.output_path}/{args.dataset}_{args.timesplit}_winlen_{args.window_len}_stepsize_{args.step_size}_{args.preprocess}'
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare the data
    df_data = pd.read_csv(args.data_path)
    data = preprocess(args, df_data[args.features])
    
    # Generate predictions
    results = {}
    drift_methods = args.models
    for model in drift_methods:
        df_preds = pd.DataFrame()
        for feature in args.features:
            df_pred = moving_prediction(args, model, data[feature], feature)
            df_preds = pd.concat([df_preds, df_pred], axis=1)
            
        num_rows = len(df_preds)
        starts = list(range(0, num_rows * args.step_size, args.step_size))
        ends = [start + args.window_len for start in starts]
        
        if len(starts) > num_rows:
            starts = starts[:num_rows]
            ends = ends[:num_rows]
            
        df_preds['start'] = starts
        df_preds['end'] = ends
        
        filename = f'{model}_{args.dataset}_{args.timesplit}_winlen_{args.window_len}_stepsize_{args.step_size}_{args.preprocess}.csv'
        df_preds.to_csv(os.path.join(output_path, filename), index=False)
        results[model] = df_preds
    
        if args.save_plot:
            print(f"Saving plots for {model}...")
            save_plots(args, output_path, df_preds, model)
    
    postprocess(args, results, output_path)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    with open('./config/config.yaml', 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
        
    for key, value in config_data.items():
        parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} argument from YAML')
    args = parser.parse_args()
    
    main(args)
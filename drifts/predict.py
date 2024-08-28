import pandas as pd
import numpy as np

from scipy.spatial import distance
from alibi_detect.cd import KSDrift, CVMDrift

def normalize_distribution(p):
    total = np.sum(p)
    return p / total

def kl_divergence(p, q):
    
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    return np.sum(p * np.log(p / q))

def process_preds(model_name, preds, feature, top_k):
    
    if model_name == 'ks' or model_name == 'cvm':
        rows = [(record['data']['distance'][0], record['data']['p_val'][0]) for record in preds]
        preds = pd.DataFrame(rows, columns=[f'{feature}_{model_name}_distance', f'{feature}_{model_name}_p_val'])
        preds[f'{feature}_{model_name}_label'] = 0
        sorted_df = preds.sort_values(by=f'{feature}_{model_name}_p_val')
    elif model_name == 'js':
        preds = pd.DataFrame(preds, columns=[f'{feature}_{model_name}_distance'])
        preds[f'{feature}_{model_name}_label'] = 0
        sorted_df = preds.sort_values(by=f'{feature}_{model_name}_distance', ascending=False)
    elif model_name == 'kl':
        preds = pd.DataFrame(preds, columns=[f'{feature}_{model_name}_distance'])
        preds[f'{feature}_{model_name}_label'] = 0
        sorted_df = preds.sort_values(by=f'{feature}_{model_name}_distance', ascending=False)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
    
    lowest_ten_indices = sorted_df.head(top_k).index
    highest_ten_indices = sorted_df.tail(top_k).index
    preds.loc[lowest_ten_indices, f'{feature}_{model_name}_label'] = 1
    preds.loc[highest_ten_indices, f'{feature}_{model_name}_label'] = 2
    
    return preds

def predict(args, model_name, data_ref, x):
    
    if model_name == 'ks':
        cd = KSDrift(data_ref, p_val=args.p_val_threshold)
        pred = cd.predict(x, drift_type='feature', return_p_val=True, return_distance=True)
    elif model_name == 'cvm':
        cd = CVMDrift(data_ref, p_val=args.p_val_threshold)
        pred = cd.predict(x, drift_type='feature', return_p_val=True, return_distance=True)
    elif model_name == 'js':
        pred = distance.jensenshannon(data_ref, x)
    elif model_name == 'kl':
        pred = kl_divergence(data_ref, x)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
    
    return pred


def moving_prediction(args, model_name, data, feature):
    
    window_len = args.window_len
    step_size = args.step_size
    ref_start, ref_end = 0, window_len
    start, end = window_len, window_len + window_len
    preds = []
    
    while end < len(data):
        data_ref = data[ref_start:ref_end].to_numpy()
        x = data[start:end].to_numpy()
        
        pred = predict(args, model_name, data_ref, x)
        preds.append(pred)
        
        ref_start += step_size
        ref_end = ref_start + window_len
        start += step_size
        end = start + window_len
    
    df_preds = process_preds(model_name, preds, feature, args.top_k)
    
    return df_preds
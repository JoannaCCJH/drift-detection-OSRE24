import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import *

def filter_signal_fft(signal, threshold=1e8):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

def filter_signal_mean(signal, window_size=5):
    
    rolling_mean = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    filtered_signal = np.concatenate([signal[:window_size-1], rolling_mean])
    
    return filtered_signal

def preprocess(args, data):
    
    filtered_data = pd.DataFrame(index=data.index)
    
    for column in data.columns:
        if args.preprocess == 'none':
            filtered_data[column] = data[column]  # No preprocessing, just copy the data
        elif args.preprocess == 'fft':
            filtered_data[column] = filter_signal_fft(data[column].values, threshold=40000)
        elif args.preprocess == 'mean':
            filtered_data[column] = filter_signal_mean(data[column].values, window_size=5)
        else:
            raise ValueError(f"Unknown preprocessing method: {args.preprocess}")
        
    return filtered_data

def save_plots(args, output_path, df_preds, model_name):
    
    df_data = pd.read_csv(args.data_path)
    
    if df_preds.columns.duplicated().any():
        raise ValueError("Duplicate column names found in df_preds.")
    
    for feature in args.features:
        
        label_column = f'{feature}_{model_name}_label'
        
        for label in [1, 2]:
            label_df = df_preds[df_preds[label_column] == label]

            fig, axs = plt.subplots(args.top_k // 2, 2, figsize=(15, 4 * (args.top_k // 2)))
            fig.suptitle(f'{feature} - Label {label}')
            axs = axs.ravel()  # Flatten the 2D array of axes

            for i, (start, end) in enumerate(zip(label_df['start'], label_df['end'])):
                if i >= args.top_k:
                    break  # Limit to 10 subplots
                
                # Slicing df_data based on start and end indices
                sliced_data = df_data.iloc[start:end]
                feature_data = sliced_data[feature]

                # Plotting the feature data range
                axs[i].plot(range(start, end), feature_data, label=f'{feature} data')
                axs[i].set_title(f'Start: {start}, End: {end}')
                axs[i].legend()

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save the figure
            save_path = f"{output_path}/{feature}_label_{label}_{model_name}_{args.preprocess}.png"
            plt.savefig(save_path)
            plt.close()  # Close the figure after saving to free up memory

            print(f"Plot saved to {save_path}")
    
    return

def postprocess(args, results, output_path):
    
    vote_results_1 = {}
    vote_results_2 = {}
    
    for feature in args.features:
        vote_results_1[feature] = pd.Series(0, index=next(iter(results.values())).index)
        vote_results_2[feature] = pd.Series(0, index=next(iter(results.values())).index)
    
    for model_name, df_preds in results.items():
        
        label_columns = [col for col in df_preds.columns if 'label' in col]
    
        for col in label_columns:
            feature = '_'.join(col.split('_')[:-2])
            if feature in vote_results_1:
                vote_results_1[feature] += (df_preds[col] == 1).astype(int)
                vote_results_2[feature] += (df_preds[col] == 2).astype(int)
            
    final_results = pd.DataFrame(index=next(iter(results.values())).index)
    for feature in vote_results_1.keys():
        final_results[feature] = 0
        final_results.loc[vote_results_1[feature] >= 2, feature] = 1
        final_results.loc[vote_results_2[feature] >= 2, feature] = 2
        # final_results[feature][vote_results_1[feature] >= 2] = 1
        # final_results[feature][vote_results_2[feature] >= 2] = 2
        
    sample_df_preds = results[next(iter(results))]  # Get the first model's df_preds
    final_results['start'] = sample_df_preds['start']
    final_results['end'] = sample_df_preds['end']
    
    final_results = adjust_labels(final_results)
    plot_final_results(args, output_path, final_results)
    
    final_results.to_csv(f'{output_path}/final_results.csv', index=False)
    
    print("Statistics:")
    for feature in args.features:
        count_1 = final_results[feature].value_counts().get(1, 0)
        count_2 = final_results[feature].value_counts().get(2, 0)
        total_rows = len(final_results)
        
        print(f"Feature: {feature}")
        print(f"  Count of label 1: {count_1}")
        print(f"  Count of label 2: {count_2}")
        print(f"  Total rows: {total_rows}")
        print()
        
    
def adjust_labels(final_results):
    
    def adjust_row(row):
        count_1 = (row == 1).sum()
        count_2 = (row == 2).sum()
        
        if count_1 > 0 and count_2 > 0:
            if count_1 > count_2:
                row[row == 1] = 2
            elif count_2 > count_1:
                row[row == 2] = 1
            else:
                row[(row == 1) | (row == 2)] = 0
        elif count_1 + count_2 == 1:
            row[(row == 1) | (row == 2)] = 0
        return row
    
    return final_results.apply(adjust_row, axis=1)
    
def plot_final_results(args, output_path, final_results):
    
    df_data = pd.read_csv(args.data_path)
    
    for feature in args.features:
        
        for label in [1, 2]:
            
            label_df = final_results[final_results[feature] == label]
            imgs_to_plot = min(10, len(label_df))
            imgs_to_plot = imgs_to_plot - 1 if imgs_to_plot % 2 != 0 else imgs_to_plot
            
            if imgs_to_plot == 0:
                continue

            fig, axs = plt.subplots(imgs_to_plot // 2, 2, figsize=(15, 4 * (imgs_to_plot // 2)))
            fig.suptitle(f'{feature} - Label {label}')
            axs = axs.ravel()  # Flatten the 2D array of axes

            for i, (start, end) in enumerate(zip(label_df['start'], label_df['end'])):
                
                if i >= imgs_to_plot:
                    break
                
                # Slicing df_data based on start and end indices
                sliced_data = df_data.iloc[start:end]
                feature_data = sliced_data[feature]

                # Plotting the feature data range
                axs[i].plot(range(start, end), feature_data, label=f'{feature} data')
                axs[i].set_title(f'Start: {start}, End: {end}')
                axs[i].legend()

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save the figure
            save_path = f"{output_path}/{feature}_label_{label}_final.png"
            plt.savefig(save_path)
            plt.close()  # Close the figure after saving to free up memory

            print(f"Final plot saved to {save_path}")
    
    return
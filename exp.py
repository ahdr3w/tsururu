import warnings
import importlib
import sys
import os
import argparse
import itertools
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tsururu.dataset import Pipeline, TSDataset
from tsururu.model_training.validator import HoldOutValidator
from tsururu.models.torch_based import NaiveEstimator
from tsururu.strategies import MIMOStrategy
from tsururu.transformers import StandardScalerTransformer
from tsururu.model_training.trainer import DLTrainer

import contextlib


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Suppress warnings
warnings.filterwarnings("ignore")

# Torch precision setup
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Script configuration")

    parser.add_argument('--output_csv', type=str, default="results_table.csv",
                        help='Path to the results CSV file')

    # Data parameters
    parser.add_argument('--df_path', type=str, default="datasets/global/simulated_data_to_check.csv",
                        help='Path to the dataset CSV file')
    parser.add_argument('--id', type=int, default=0,
                        help='Id of device')
    parser.add_argument('--train_size', type=float, default=0.7,
                        help='Size of the training set')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Size of the test set')
    parser.add_argument('--history', type=int, default=336,
                        help='History length')
    parser.add_argument('--n_nodes', type=int, default=10,
                        help='Number of nodes')

    # Model parameters
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')
    parser.add_argument('--node_dim', type=int, default=40,
                        help='Node dimensions')
    parser.add_argument('--hidden_dim', type=int, default=384,
                        help='Hidden dimensions')
    parser.add_argument('--node_proj', type=bool, default=True,
                        help='Node projection')
    parser.add_argument('--nonlinear_pred', type=bool, default=True,
                        help='Nonlinear prediction')
    parser.add_argument('--norm', type=bool, default=True,
                        help='Normalization')
    parser.add_argument('--mix', type=bool, default=True,
                        help='Mixing')
    parser.add_argument('--adj', type=str, default="multihead_attention",
                        help='Adjacency type')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of heads')
    parser.add_argument('--k', type=int, default=4,
                        help='K value for adjacency')
    parser.add_argument('--dim', type=int, default=96,
                        help='Dimensions')
    parser.add_argument('--alpha', type=int, default=3,
                        help='Alpha value')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--leaky_rate', type=float, default=0.2,
                        help='Leaky rate')

    # Trainer parameters
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers')
    parser.add_argument('--stop_by_metric', type=bool, default=True,
                        help='Stop by metric')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0,
                        help='Weight decay')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='Learning rate factor')
    parser.add_argument('--n_warmup', type=int, default=5,
                        help='Number of warmup steps')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Gamma for learning rate scheduler')

    return parser.parse_args()

# Get command line arguments
args = parse_args()

def get_lr_lambda(factor, n_warmup, gamma):
    def lr_lambda(n):
        if n < n_warmup:
            return factor + (1 - factor) / (n_warmup - 1) * n
        else:
            return gamma * (n - n_warmup + 1)
    return lr_lambda

def generate_param_grid(param_options):
    keys, values = zip(*param_options.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments

# Scheduler parameters
lr_lambda = get_lr_lambda(factor=args.factor, n_warmup=args.n_warmup, gamma=args.gamma)

# Now, let's continue with the rest of your script

def get_results(cv: int, regime: str, y_true: Optional[List[np.ndarray]] = None, y_pred: Optional[List[np.ndarray]] = None, ids: Optional[List[Union[float, str]]] = None) -> pd.DataFrame:
    def _get_fold_value(value: Optional[Union[float, np.ndarray]], idx: int) -> List[Optional[Union[float, np.ndarray]]]:
        if value is None:
            return [None]
        if isinstance(value[idx], float):
            return value[idx]
        if isinstance(value[idx], np.ndarray):
            return value[idx].reshape(-1)
        raise TypeError(f"Unexpected value type. Value: {value}")

    df_res_dict = {}

    for idx_fold in range(cv):
        # Fill df_res_dict
        for name, value in [("y_true", y_true), ("y_pred", y_pred)]:
            df_res_dict[f"{name}_{idx_fold+1}"] = _get_fold_value(value, idx_fold)
        if regime != "local":
            df_res_dict[f"id_{idx_fold+1}"] = _get_fold_value(ids, idx_fold)

    # Save datasets to specified directory
    df_res = pd.DataFrame(df_res_dict)
    return df_res

def expand_val_with_train(train_data, val_data, id_column, date_column, history):
    L_split_data = train_data[date_column].values[(len(train_data) - history)]
    L_last_train_data = train_data[train_data[date_column] >= L_split_data]
    val_data_expanded = pd.concat((L_last_train_data, val_data))
    val_data_expanded = val_data_expanded.sort_values([id_column, date_column]).reset_index(drop=True)
    return val_data_expanded

def expand_test_with_val_and_train(train_data, val_data, test_data, id_column, date_column, history):
    unqiue_id_cnt = val_data[id_column].nunique()
    L_split_data = val_data[date_column].values[((len(val_data) - history) if (len(val_data) // val_data[id_column].nunique() - history) > 0 else 0)]
    L_last_val_data = val_data[val_data[date_column] >= L_split_data]
    if len(val_data) // unqiue_id_cnt - history < 0:
        if (len(train_data) - (history - len(L_last_val_data) / unqiue_id_cnt)) > 0:
            L_split_data = train_data[date_column].values[(len(train_data) // unqiue_id_cnt - (history - len(L_last_val_data) // unqiue_id_cnt))]
        else:
            L_split_data = 0
        L_last_train_data = train_data[train_data[date_column] >= L_split_data]
        test_data_expanded = pd.concat((L_last_train_data, L_last_val_data, test_data))
    else:
        test_data_expanded = pd.concat((L_last_val_data, test_data))
    test_data_expanded = test_data_expanded.sort_values([id_column, date_column]).reset_index(drop=True)
    return test_data_expanded

def get_train_val_test_datasets(dataset_path, columns_params, train_size, test_size, history):
    data = pd.read_csv(dataset_path)
    date_column = columns_params["date"]["columns"][0]
    id_column = columns_params["id"]["columns"][0]

    if dataset_path.endswith(tuple(["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"])):
        train_val_split_data = "2017-06-25 23:00:00"
        val_test_slit_data = "2017-10-23 23:00:00"
    else:
        train_val_split_data = data[date_column].values[int(data[date_column].nunique() * train_size)]
        val_test_slit_data = data[date_column].values[int(data[date_column].nunique() * (1 - test_size))]

    train_data = data[data[date_column] <= train_val_split_data]
    val_data = data[(data[date_column] > train_val_split_data) & (data[date_column] <= val_test_slit_data)]
    test_data = data[data[date_column] > val_test_slit_data]
    val_data = expand_val_with_train(train_data, val_data, id_column, date_column, history)
    test_data_expanded = expand_test_with_val_and_train(train_data, val_data, test_data, id_column, date_column, history)

    train_dataset = TSDataset(data=train_data, columns_params=columns_params)
    val_dataset = TSDataset(data=val_data, columns_params=columns_params)
    test_dataset = TSDataset(data=test_data_expanded, columns_params=columns_params)

    return train_dataset, val_dataset, test_dataset

def scale(df, pipeline):
    def get_scaler(obj):
        if isinstance(obj, StandardScalerTransformer):
            return obj
        if "transformers" in dir(obj):
            return get_scaler(obj.transformers)
        if "transformers_list" in dir(obj):
            curr_obj = None
            for tf in obj.transformers_list:
                curr_obj = get_scaler(tf)
                if isinstance(curr_obj, StandardScalerTransformer):
                    return curr_obj
        return None

    scaler = get_scaler(pipeline)
    if scaler is not None:
        new_df = pd.DataFrame(columns=df.columns)
        for i in df["id"].unique():
            temp = df[df["id"] == int(i)].copy()
            temp = scaler._transform_segment(temp, "id")
            temp = temp.drop("value", axis=1)
            temp = temp.rename({"value__standard_scaler": "value"}, axis=1)
            new_df = pd.concat((new_df, temp), axis=0)
        return new_df
    else:
        return df

def exp():
    columns_params = {
        "target": {
            "columns": ["value"],
            "type": "continious",
        },
        "date": {
            "columns": ["date"],
            "type": "datetime",
        },
        "id": {
            "columns": ["id"],
            "type": "categorical",
        }
    }
    
    
    param_options = {
        "adj": ["identity", "linear", "global", "directed", "undirected", "unidirected", 
                "graph_wavenet", "dynamic_attention", "multihead_attention"],

        "node_proj": [False, True],
        "nonlinear_pred": [False, True],
        
    }

    experiments = generate_param_grid(param_options)
    
    results = []

    for params in tqdm(experiments, leave=False):

        train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(args.df_path, columns_params, args.train_size, args.test_size, args.history)
    
        pipeline_params = {
            "target": {
                "columns": ["value"],
                "features": {
                    "StandardScalerTransformer": {
                        "transform_target": True,
                        "transform_features": True,
                    },
                    "LagTransformer": {"lags": args.history},
                },
            },
        }

        validation = HoldOutValidator
        validation_params = {"validation_data": val_dataset}
        
        pipeline = Pipeline.from_dict(pipeline_params, multivariate=True)


        model_params = {
            "seq_len": args.history,
            "pred_len": args.pred_len,
            "hidden_dim": 4*args.history,
            "attn_len": args.history,

            "node_proj": params['node_proj'],
            "nonlinear_pred": params['nonlinear_pred'],
            "adj": params['adj'],

            "mix": args.mix,
            "node_dim": 4*args.n_nodes,
            "norm": args.norm,
            "num_heads": args.num_heads,
            "k": args.n_nodes,
            "dim": 4*args.n_nodes,
            "alpha": args.alpha,
            
            "enc_in": args.n_nodes,
            "n_nodes": args.n_nodes,
            "static_feat": None,
            "leaky_rate": args.leaky_rate,
            "dropout_rate": args.dropout_rate,
        }
        
        trainer_params = {
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "num_workers": args.num_workers,
            "stop_by_metric": args.stop_by_metric,
            "patience": args.patience,
            "optimizer": torch.optim.AdamW,
            "optinizer_params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": torch.optim.lr_scheduler.LambdaLR,
            "scheduler_params": {"lr_lambda": lr_lambda},
            "checkpoint_path": f"checkpoints/{args.id}/"
        }
        
        mae_list = []
        mse_list = []

        # Running the experiment several times to get mean and std
        for _ in range(5):  # Adjust the number of repetitions as needed
            trainer = DLTrainer(NaiveEstimator, model_params, validation, validation_params, **trainer_params)
            strategy = MIMOStrategy(pipeline=pipeline, trainer=trainer, horizon=args.pred_len, history=args.history, step=1)
            
            
            with suppress_output():
                strategy.fit(train_dataset)
            with suppress_output():
                pred = strategy.predict(test_dataset, test_all=True)[1]

            true = test_dataset.data
            merged_df = pd.merge(pred, true, on=['id', 'date'], suffixes=('_df1', '_df2'))

            true = merged_df[['id', 'date', 'value_df2']].rename(columns={'value_df2': 'value'})
            true_scaled = scale(true, pipeline)
            pred_scaled = scale(pred, pipeline)

            mae = np.abs(pred_scaled["value"].values - true_scaled["value"].values).mean()
            mse = np.square(pred_scaled["value"].values - true_scaled["value"].values).mean()
            
            mae_list.append(mae)
            mse_list.append(mse)
        
        mae_mean = np.mean(mae_list)
        mae_std = np.std(mae_list)
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list)

        results.append({
            'model': 'NaiveEstimator',

            'dataset': args.df_path.split('/')[-1].split('.')[0],
            'enc_in': args.n_nodes,
            'history': args.history,
            'horizon': args.pred_len,

            "project_nodes": params['node_proj'],
            "predict_nonlinear": params['nonlinear_pred'],
            "adj": params['adj'],

            'mae_mean': mae_mean,
            'mae_std': mae_std,
            'mse_mean': mse_mean,
            'mse_std': mse_std,
        })


        print(f"Parameters: {params}", end='\r', flush=True)
        print(f"Results: MAE={mae_mean:.3f} ± {mae_std:.3f}, MSE={mse_mean:.3f} ± {mse_std:.3f}", end='\r', flush=True)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"{args.output_csv}_{args.df_path.split('/')[-1].split('.')[0]}", index=False)
        print(f"Results saved to {args.output_csv}_{args.df_path.split('/')[-1].split('.')[0]}", end='\r', flush=True)

if __name__ == "__main__":
    exp()
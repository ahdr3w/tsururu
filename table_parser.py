import pandas as pd

def parse(file_path):
    lines = ''
    with open(file_path, "r+") as f:
        lines = f.read()
        lines = lines.replace("_", " ")
        
    with open(file_path, 'w') as f:
        f.write(lines)
    df = pd.read_csv(file_path)

    # Create LaTeX table
    datasets = df['dataset'].unique()
    adjs = df['adj'].unique()

    result = ""

    # Start table
    result += "\\begin{table}[h!]\n\centering\n\\begin{tabular}{|l|"  + "p{1.7cm}|" * len(datasets) * 2 + "}\n\hline\n"

    str_datasets = [f"\multicolumn{'{2}'}{'{c|}'}{'{'+dataset+'}'}" for dataset in datasets]

    # Header
    result += "Model & " + ' & '.join(str_datasets) + f" \\\\\n\cline{'{2-'+f'{2*len(datasets)+1}'+'}'}\n"

    result += " & MAE & MSE" * len(datasets) + " \\\\\n\hline\n"

    # Fill table
    for adj in adjs:
        # Base row for adjacency
        mae_means = []
        mse_means = []
        for dataset in datasets:
            sub_df = df[(df['adj'] == adj) & (df['dataset'] == dataset)]
            row = sub_df[(sub_df['project nodes'] == False) & (sub_df['predict nonlinear'] == False)].iloc[0]
            mae_means.append(f"{row['mae mean']:.3f}±{row['mae std']:.3f}")
            mse_means.append(f"{row['mse mean']:.3f}±{row['mse std']:.3f}")

        result += f"{adj} & "
        result += ' & '.join([f'{mae} & {mse}' for mae, mse in zip(mae_means, mse_means)])
        result += " \\\\\n"
    

        # Row for nonlinearity in feature domain
        mae_means = []
        mse_means = []
        for dataset in datasets:
            sub_df = df[(df['adj'] == adj) & (df['dataset'] == dataset)]
            row = sub_df[(sub_df['project nodes'] == False) & (sub_df['predict nonlinear'] == True)].iloc[0]
            mae_means.append(f"{row['mae mean']:.3f}±{row['mae std']:.3f}")
            mse_means.append(f"{row['mse mean']:.3f}±{row['mse std']:.3f}")

        result += f"+ nonlin. projection & "
        result += ' & '.join([f'{mae} & {mse}' for mae, mse in zip(mae_means, mse_means)])
        result += " \\\\\n"
        

        # Row for prediction nonlinearity
        mae_means = []
        mse_means = []
        for dataset in datasets:
            sub_df = df[(df['adj'] == adj) & (df['dataset'] == dataset)]
            row = sub_df[(sub_df['project nodes'] == True) & (sub_df['predict nonlinear'] == False)].iloc[0]
            mae_means.append(f"{row['mae mean']:.3f}±{row['mae std']:.3f}")
            mse_means.append(f"{row['mse mean']:.3f}±{row['mse std']:.3f}")

        result += f"+ nonlin. features & "
        result += ' & '.join([f'{mae} & {mse}' for mae, mse in zip(mae_means, mse_means)])
        result += " \\\\\n"
        

        mae_means = []
        mse_means = []
        for dataset in datasets:
            sub_df = df[(df['adj'] == adj) & (df['dataset'] == dataset)]
            row = sub_df[(sub_df['project nodes'] == True) & (sub_df['predict nonlinear'] == True)].iloc[0]
            mae_means.append(f"{row['mae mean']:.3f}±{row['mae std']:.3f}")
            mse_means.append(f"{row['mse mean']:.3f}±{row['mse std']:.3f}")

        result += f"+ both & "
        result += ' & '.join([f'{mae} & {mse}' for mae, mse in zip(mae_means, mse_means)])
        result += " \\\\\n"
        result += "\hline\n"

    # End table

    result += "\end{tabular}\n\caption{Results summary}\n\end{table}\n"

    # Print the result
    print(result)
    with open("parsed_table.txt", 'w') as f:
        f.write(result)


if __name__ == "__main__":
    file_path = 'results_table.csv'  
    parse(file_path)


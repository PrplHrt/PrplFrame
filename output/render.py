import os
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import numpy as np

def render_results_html(dataset_info: dict, scores: list[dict], directory: str = None):
    """
    Function that uses jinja and a saved template to create an HTML page with all the results of this experiment.
    Expects a dataset_info dictionary and a list of scores, each with the following structures:
    dataset_info = {
        'name':...,
        'type':...,
        'target':...,
        'size':...,
        'split':...,
        'source':...
        }
    scores = [
        {
            'name':...,
            'r2':...,
            'mse':...,
            'train_time':...,
            'test_time:...
        },
        {
        ...
        },
        ...
        ]
    
    """
    environment = Environment(loader=FileSystemLoader("output/templates/"))
    results_template = environment.get_template("regression_results.html")

    context = {
        'dataset_info': dataset_info,
        'scores': scores
    }

    if directory:
        os.makedirs(directory, exist_ok=True)
        results_filename = os.path.join(directory, f'{dataset_info["name"]}_results.html')
    else:
        os.makedirs('results', exist_ok=True)
        results_filename = f'results/{dataset_info["name"]}_results.html'
    
    with open(results_filename, mode="w", encoding="utf-8") as results:
        results.write(results_template.render(context))
        print(f"Results page for {dataset_info['name']} saved in {results_filename}...")
    
    return results_filename

def plot_helper(x: np.ndarray, y: np.ndarray, column : str, target: str, stats: pd.DataFrame, save_dir: str):
    # Create a wider figure to accommodate the graph and text box side by side
    plt.figure(figsize=(12, 6))

    # Plot the graph on the left side
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(x, y)
    plt.title(f"Parametric Study - {textwrap.fill(target + ' x ' + column, width=30)}")
    plt.xlabel(column)
    plt.ylabel(target)


    # Add a box of values on the right side
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    box_text = "Base Values:\n" + "\n\n".join([f"{textwrap.fill(str(index) + ': ' + str(row.Mean), width=40)}" for index, row in stats.iterrows()])
    plt.text(0.1, 0.5, box_text, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8), fontsize=15, va='center')

    # Remove axes from the second subplot
    ax = plt.gca()
    ax.axis('off')

    # Adjust spacing between subplots to avoid overlapping
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"{target}_{column}.png"))
    plt.close()

def plot_parametric_graphs(stats: pd.DataFrame, results: dict, target: str | list[str], directory: str = "", make_excel: bool = False):
    save_dir = os.path.join(directory, "parametric")
    os.makedirs(save_dir,exist_ok=True)
    if make_excel:
        sheets = []
    for column, vals in results.items():
        x, y = vals

        if type(target) == str:
            plot_helper(x, y, column, target, stats, save_dir)
            if make_excel:
                df = stats.transpose().drop(column, axis=1).drop(['Max', 'Min'])
                df = pd.concat([df]*len(x), ignore_index=True)
                df[column] = x
                df[target] = y
                sheets.append([column, df])
        else:
            if make_excel:
                df = stats.transpose().drop(column, axis=1).drop(['Max', 'Min'])
                df = pd.concat([df]*len(x), ignore_index=True)
                df[column] = x
            for i, targ in enumerate(target):
                temp_y = [val[i] for val in y]
                plot_helper(x, temp_y, column, targ, stats, save_dir)
                if make_excel:
                    df[targ] = temp_y
            if make_excel:
                sheets.append([column, df])
    
    print("Parametric plots saved in directory: " + save_dir)
    if make_excel:
        with pd.ExcelWriter(os.path.join(save_dir, 'parametric_data.xlsx')) as writer:
            for sheet in sheets:
                sheet[1].to_excel(writer, sheet_name=sheet[0], index=False)
        print("Parametric plots data saved in directory: ", os.path.join(save_dir, 'parametric_data.xlsx'))
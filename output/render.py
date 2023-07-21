import os
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

def render_results_html(dataset_info: dict, scores: list[dict], filename: str = None):
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

    if filename:
        results_filename = filename
    else:
        os.makedirs('results', exist_ok=True)
        results_filename = f'results/{dataset_info["name"]}_results.html'
    
    with open(results_filename, mode="w", encoding="utf-8") as results:
        results.write(results_template.render(context))
        print(f"Results page for {dataset_info['name']} saved in {results_filename}...")
    
    return results_filename

def plot_parametric_graphs(stats: pd.DataFrame, results: dict, target: str, directory: str = ""):
    save_dir = os.path.join(directory, "parametric")
    os.makedirs(save_dir,exist_ok=True)
    for column, vals in results.items():
        x, y = vals

         # Create a wider figure to accommodate the graph and text box side by side
        plt.figure(figsize=(12, 6))

        # Plot the graph on the left side
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.plot(x, y)
        plt.title(f"Parametric Study - {textwrap.fill(column + ' x ' + target, width=30)}")
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

        plt.savefig(os.path.join(save_dir, f"{column}.png"))
        plt.close()
    print("Parametric plots saved in directory: " + save_dir)
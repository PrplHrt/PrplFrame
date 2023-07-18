import os
from jinja2 import Environment, FileSystemLoader

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

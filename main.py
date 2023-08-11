from typing import Iterable
import pandas as pd

from evaluation import utils
from models import regression
from output import render
import os

# Dataset info
dataset_info = {'name': 'Concrete Compressive Strength 2',
                'type': 'Regression',
                'target': 'Concrete compressive strength(MPa, megapascals) ',
                'split': 0.2,
                'path': "Concrete_Data 2.xls",
                'source': """Prof. I-Cheng Yeh
  Department of Information Management 
  Chung-Hua University, 
  Hsin Chu, Taiwan 30067, R.O.C.
  e-mail:icyeh@chu.edu.tw
  TEL:886-3-5186511"""}

models = [regression.GaussianProcessRegressor(),
          regression.KNeighborsRegressor(),
          regression.Ridge(),
          regression.LinearRegression(),
          regression.MLPRegressor(),
          regression.PolynomialRegression(),
          regression.SVR(),
          regression.DecisionTreeRegressor(),
          regression.Lasso(),
          regression.RandomForestRegressor()
          ]

dataset_dir = "data/"

def main():
    # Test with 1 target
    dataset = pd.read_excel(dataset_dir + dataset_info['path'])
    dataset_info['size'] = len(dataset)

    data = utils.load_Xy(dataset, dataset_info['target'], dataset_info['split'])

    # Using a temporary variable to hold the best performing model
    # we'll define the best performing model by the highest r2
    top_model = None
    best_r2 = None

    scores=[]
    for model in models:
        score = utils.regression_train_and_test(model, *data)
        # Check for best R2 score
        if (not best_r2) or (score['r2'] > best_r2):
            top_model = model
            best_r2 = score['r2']
        scores.append(score)

    scores.sort(key=lambda x: x['mse'])

    render.render_results_html(dataset_info, scores)

    stats, results = utils.regression_parametric_study(top_model, dataset, dataset_info['target'], c1=[0.1, 0.2, 0.3, 0.4, 0.5], c2=[6, 9, 12, 15, 18, 21, 24, 27, 30, 33])
    render.plot_parametric_graphs(stats, results, dataset_info['target'], 'results', True)

    os.makedirs(os.path.join('results', "custom_parametric"), exist_ok=True)
    results, values = utils.custom_parametric(top_model, dataset,{dataset.columns[1]: range(0, 10), dataset.columns[2]: range(9, 12)}, dataset_info['target'])
    df = pd.DataFrame(values, columns=dataset.drop(dataset_info['target'], axis=1).columns)
    df[dataset_info['target']] = results
    df.to_csv(os.path.join('results', "custom_parametric", 'custom_parametric_data.csv'))

def two_target():
    # Test with 2 targets
    dataset_info['target'] = ['Concrete compressive strength(MPa, megapascals) ', 'Age (day)']
    dataset = pd.read_excel(dataset_dir + dataset_info['path'])
    dataset_info['size'] = len(dataset)

    data = utils.load_Xy(dataset, dataset_info['target'], dataset_info['split'])

    # Using a temporary variable to hold the best performing model
    # we'll define the best performing model by the highest r2
    top_model = None
    best_r2 = None

    scores=[]
    for model in models:
        score = utils.regression_train_and_test(model, *data)
        # Check for best R2 score
        if (not best_r2) or (score['r2'] > best_r2):
            top_model = model
            best_r2 = score['r2']
        scores.append(score)

    scores.sort(key=lambda x: x['mse'])

    render.render_results_html(dataset_info, scores, f"results/twoTargets")
    
    stats, results = utils.regression_parametric_study(top_model, dataset, dataset_info['target'])

    render.plot_parametric_graphs(stats, results, dataset_info['target'], "results/twoTargets", True)

if __name__ == "__main__":
    main()
    # two_target()
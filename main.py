import pandas as pd

from evaluation import utils
from models import regression
from output import render

dataset_info = {'name': 'Concrete Compressive Strength',
                'type': 'Regression',
                'target': 'Concrete compressive strength(MPa, megapascals) ',
                'split': 0.2,
                'path': "Concrete_Data.xls",
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
    dataset = pd.read_excel(dataset_dir + dataset_info['path'])
    dataset_info['size'] = len(dataset)

    data = utils.load_Xy(dataset, dataset_info['target'], dataset_info['split'])

    scores=[]
    for model in models:
        scores.append(utils.regression_train_and_test(model, *data))

    scores.sort(key=lambda x: x['mse'])

    render.render_results_html(dataset_info, scores, f"results/{dataset_info['name']}_results.html")

if __name__ == "__main__":
    main()
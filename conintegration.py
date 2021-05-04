import pandas as pd
import datetime 
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller


def get_adf_test_results(x, y, start_date, end_date):
    """
    Tests conintegration relationship

    param: x: independent variable;
           y: dependent variable;
           start_date: start date of the backtest period
           end_date: end date of the backtest period
    """
    x = pd.read_csv(x, index_col = 'Date')['Adj Close'][start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].to_numpy().reshape(-1, 1)
    y = pd.read_csv(y, index_col = 'Date')['Adj Close'][start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].to_numpy().reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    s = y - y_pred
    result = adfuller(s)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Significance level: ')
    for key, value in result[4].items():
        print('\t{}: {}'. format(key, value))
    if result[1] <= 0.05:
        print('x and y are conintegrated!')


if __name__ == '__main__':
    start_date = datetime.datetime(2011, 4, 29)
    end_date = datetime.datetime(2021, 4, 30)
    x = 'data/UPRO.csv'
    y = 'data/VOO.csv'
    get_adf_test_results(x, y, start_date, end_date)
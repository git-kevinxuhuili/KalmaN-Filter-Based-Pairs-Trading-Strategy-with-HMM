from __future__ import print_function

import datetime 
import pickle 
import warnings 
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import numpy as np 
import pandas as pd 
import seaborn as sns 


def obtain_prices_df(csv_filepath, start_date, end_date):
    """
    Obtains the prices DataFrame from the CSV file, 
    filter by the end date and calculate the 
    percentage returns. 
    """
    df = pd.read_csv(
        csv_filepath, header = 0,
        names = [
            "Date", "Open", "High", "Low",
            "Close", "Volume", "Adj Close"
        ],
        index_col = "Date", parse_dates=True
    )
    df['Returns'] = df['Adj Close'].pct_change()
    df = df[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
    df.dropna(inplace = True)
    return df 


def plot_in_sample_hidden_states(hmm_model, df):
    """
    Plots the adjusted closing prices masked by 
    the in-sample hidden states as a mechanism 
    to understand the market regimes.
    """
    # Predict the hidden states array 
    hidden_states = hmm_model.predict(rets)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        hmm_model.n_components,
        sharex = True, sharey = True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df.index[mask],
            df["Adj Close"][mask],
            ".", linestyle = 'none',
            c = colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()


if __name__ == '__main__':

    # Hides deprecation warnings for sklearn
    warnings.filterwarnings('ignore')

    csv_filepath = "/Users/xuhuili/Desktop/ST451_Bayesian_Machine_Learning/Project/data/VOO.csv"
    pickle_path = "/Users/xuhuili/Desktop/ST451_Bayesian_Machine_Learning/Project/model/hmm_model_voo.pkl"
    # csv_filepath = "/Users/xuhuili/Desktop/ST451_Bayesian_Machine_Learning/Project/data/UPRO.csv"
    # pickle_path = "/Users/xuhuili/Desktop/ST451_Bayesian_Machine_Learning/Project/model/hmm_model_upro.pkl"
    # Training period: April 30th, 2011 to April 30th, 2019
    start_date = datetime.datetime(2011, 4, 29)
    end_date = datetime.datetime(2019, 4, 29)
    asset = obtain_prices_df(csv_filepath, start_date, end_date)
    rets = np.column_stack([asset["Returns"]])

    # Shows the histogram plot for the returns
    _ = plt.hist(rets)
    plt.show()

    # Create the Gaussian Hidden Markov Model and fit it 
    # to the asset returns data, outputting a score 
    hmm_model = GaussianHMM(
        n_components = 2, covariance_type="full", n_iter=1000
    ).fit(rets)
    print('Model Score: ', hmm_model.score(rets))

    # Plot the in-sample hidden states closing values 
    plot_in_sample_hidden_states(hmm_model, asset)

    print('Picking HMM model...')
    pickle.dump(hmm_model, open(pickle_path, "wb"))
    print("...HMM model pickled.")






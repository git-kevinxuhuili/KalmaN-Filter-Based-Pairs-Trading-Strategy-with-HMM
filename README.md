# A Practical Application of Hidden Markov Model to Kalman Filter-Based Pairs Trading

A protect for the module ST451 Bayesian Mahine Learning

The Jupyter Notebook is used for presenting the backtesting results.

Please find the other four python files in the folder.

regime_hmm_train.py --- used for training the HMM model

conintegration.py --- used for testing the conintegration relationship

hmm_risk_manager.py --- HMM risk manager component

kalman_filter_strategy --- Kalman Filter Pairs Trading Strategy component


### Abstract 

The objective of this project is to implement a Bayesian updating process called the Kalman Filter in a common quantitative trading technique, which involves taking two assets that form a cointegrated relationship and utilising the mean-reverting nature between them, so called pairs trading. Furthermore, a common but difficult task for the quantitative trading participants is to detect the market regime change in the financial market. To address such challenge, we apply Hidden Markov Model for carrying out regime detection. We believe that by combining these two Bayesian approaches, a profitable while well-protected  pairs trading strategy is achievable.

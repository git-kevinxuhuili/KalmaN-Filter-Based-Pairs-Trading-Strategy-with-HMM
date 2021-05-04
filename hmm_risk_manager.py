from __future__ import print_function

import numpy as np 
from qstrader.event import OrderEvent, SignalEvent
from qstrader.price_parser import PriceParser
from qstrader.risk_manager.base import AbstractRiskManager


class HMMRiskManager(AbstractRiskManager):
    """
    Utilises a previously fitted Hidden Markov Model 
    as a market regime detector. The risk manager 
    ignores orders that occur during a non-desirable 
    regime for either asset.

    It also accounts for the fact that a trade may 
    straddle two separate regimes. If a close order 
    is received in the undesirable regime, and the 
    order is open, it will be closed, but no new 
    orders are generared until the desirable regime 
    is achieved. 
    """

    def __init__(self, hmm_model_1, hmm_model_2):
        self.hmm_model_1 = hmm_model_1
        self.hmm_model_2 = hmm_model_2
        self.invested = False


    def determine_regime(self, price_handler, sized_order):
        """
        Determines the predicted regime by making a prediction 
        on the adjusted closing returns from the price handler
        object and then taking the final entry integer as 
        the " hidden regime state".
        """
        returns = np.column_stack(
            [np.array(price_handler.adj_close_returns)]
        )
        hidden_states_1 = self.hmm_model_1.predict(returns)[-2]
        hidden_states_2 = self.hmm_model_2.predict(returns)[-1]
        # print(hidden_states_1, hidden_states_2)
        return hidden_states_1, hidden_states_2

    
    def refine_orders(self, portfolio, sized_order):
        """
        Uses the Hidden Markov Model with the percentage returns 
        to predict the current regime, either 0 for desirable or 1
        for undesirable. Long entry trades will only be carried 
        out in regime 0, but closing trades are allowed in regime 1.
        """
        # Deterime the HMM predicted regime as an integer 
        # equal to 0 (desirable) or 1 (undesirable)

        price_handler = portfolio.price_handler
        regime_1, regime_2 = self.determine_regime(
           price_handler, sized_order
        )
        action = sized_order.action
        # Create the order event, irrespective of the regime.
        # It will only be returned if the correct conditions 
        # are met below.
        order_event = OrderEvent(
            sized_order.ticker,
            sized_order.action,
            sized_order.quantity
        )
        # If in the desirable regime, let buy and sell orders 
        # works as normal
        if regime_1 == 0 and regime_2 == 0:
            print('Regime 0')
            if self.invested == False:
                print('Open')
                if action == "BOT":
                    self.invested == False
                    return [order_event]
                elif action == "SLD":
                    self.invested == True
                    return [order_event]
            elif self.invested == True:
                print('Close')
                if action == "BOT":
                    self.invested == True
                    return [order_event]
                elif action == "SLD":
                    self.invested == False
                    return [order_event]
        # If in the undesirable regime, do not allow any buy orders
        # and only let close orders through if the strategy is 
        # already invested (from a previous desirable regime)
        elif regime_1 == 1 or regime_2 == 1:
            print('Regime 1')
            if self.invested == False:
                print('Bad Market Condition, do not invest')
                if action == "BOT":
                    self.invested == False
                    return []
                elif action == "SLD":
                    self.invested == False
                    return []
            elif self.invested == True:
                print('Bad Trade, get out of the trade')
                if action == "BOT":
                    self.invested == True
                    return [order_event]
                elif action == "SLD":
                    self.invested == False
                    return [order_event]
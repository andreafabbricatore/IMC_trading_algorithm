import pandas as pd
import numpy as np

from datamodel import *
from typing import Dict, List, Optional, Any


FAIR_PRICE_PEARLS = 10000
TIME_TICK = 100

POSITION_LIMITS = {
    'PEARLS': 20,
    'BANANAS': 20,
    'PINA_COLADAS': 300,
    'COCONUTS': 600
}

CONFIG_SLACK_TRADER_PEARLS = {
    'EDGE': 2,
    'STEP': 0.4,
    'MIN_EDGE': 0.5,  # should always be less than EDGE
    'ROUNDS_THRESHOLD': 5
}

CONFIG_SLACK_TRADER_BANANAS = {
    'EDGE': 1.5,
    'STEP': 0.25,
    'MIN_EDGE': 0.5,  # should always be less than EDGE
    'ROUNDS_THRESHOLD': 10
}

CONFIG_SLACK_TRADER_PINA_COLADAS = {
    'EDGE': 1.0,
    'STEP': 0.25,
    'MIN_EDGE': 0.5,  # should always be less than EDGE
    'ROUNDS_THRESHOLD': 3
}


def position(state: TradingState, symbol: str) -> int:
    return state.position[symbol] if symbol in state.position else 0


def get_mid_price(state: TradingState, symbol: str, default=None) -> float:
    order_depth: OrderDepth = state.order_depths[symbol]
    buy_orders = order_depth.buy_orders
    sell_orders = order_depth.sell_orders
    if len(buy_orders) == 0 or len(sell_orders) == 0:
        return default
    return (max(buy_orders.keys()) + min(sell_orders.keys())) / 2.0


def get_weighted_mid_price(state: TradingState, symbol: str, default=None) -> float:
    order_depth: OrderDepth = state.order_depths[symbol]
    buy_orders = order_depth.buy_orders
    sell_orders = order_depth.sell_orders
    if len(buy_orders) == 0 or len(sell_orders) == 0:
        return default
    weighted_buy_price = sum([price * quantity for price, quantity in buy_orders.items()]) / sum(buy_orders.values())
    weighted_sell_price = sum([price * quantity for price, quantity in sell_orders.items()]) / sum(sell_orders.values())
    return (weighted_buy_price + weighted_sell_price) / 2.0


def rounds_from_last_trade(state: TradingState, symbol: str, side: str) -> Optional[int]:
    if symbol in state.own_trades:
        for trade in state.own_trades[symbol][::-1]:
            if (side == 'ASK' and trade.seller == "self") or (side == 'BID' and trade.buyer == "self"):
                return (state.timestamp - trade.timestamp) // TIME_TICK
    return 0




class Trader:
    
    def __init__(self):
        self.edge_ask = {
            'PEARLS': CONFIG_SLACK_TRADER_PEARLS['EDGE'],
            'BANANAS': CONFIG_SLACK_TRADER_BANANAS['EDGE'],
            'PINA_COLADAS': CONFIG_SLACK_TRADER_PINA_COLADAS['EDGE']
        }
        self.edge_bid = {
            'PEARLS': CONFIG_SLACK_TRADER_PEARLS['EDGE'],
            'BANANAS': CONFIG_SLACK_TRADER_BANANAS['EDGE'],
            'PINA_COLADAS': CONFIG_SLACK_TRADER_PINA_COLADAS['EDGE']
        }

    def slack_trader(self, state: TradingState,
                     config: Dict, symbol: str,
                     price_override: Optional[int] = None,
                     price_func=get_weighted_mid_price) -> List[Order]:

        no_trade_ask_rounds = rounds_from_last_trade(state, symbol, "ASK")
        if no_trade_ask_rounds > config['ROUNDS_THRESHOLD']:
            self.edge_ask[symbol] = max(config['MIN_EDGE'], self.edge_ask[symbol] - config['STEP'])
        else:
            self.edge_ask[symbol] = min(config['EDGE'], self.edge_ask[symbol] + config['STEP'])

        no_trade_bid_rounds = rounds_from_last_trade(state, symbol, "BID")
        if no_trade_bid_rounds > config['ROUNDS_THRESHOLD']:
            self.edge_bid[symbol] = max(config['MIN_EDGE'], self.edge_bid[symbol] - config['STEP'])
        else:
            self.edge_bid[symbol] = min(config['EDGE'], self.edge_bid[symbol] + config['STEP'])

        orders: list[Order] = []

        current_position = position(state, symbol)
        max_vol_ask: int = -POSITION_LIMITS[symbol] - current_position
        max_vol_bid: int = POSITION_LIMITS[symbol] - current_position

        fair_price = price_override if price_override is not None else price_func(state, symbol)
        orders.append(Order(symbol, round(fair_price - self.edge_bid[symbol]), max_vol_bid))  # bid order
        orders.append(Order(symbol, round(fair_price + self.edge_ask[symbol]), max_vol_ask))  # ask order
        return orders

    # product to trade, will be hedged with coconuts
    def trade_pinacoladas(self, state: TradingState) -> List[Order]:
        return self.slack_trader(state=state,
                                 config=CONFIG_SLACK_TRADER_PINA_COLADAS,
                                 symbol='PINA_COLADAS',
                                 price_func=get_weighted_mid_price)

    # product used to hedge the coconut position
    def trade_coconuts(self, state: TradingState) -> List[Order]:
        product = 'COCONUTS'
        orders: list[Order] = []
        curr_pos_pinacoladas = position(state, 'PINA_COLADAS')
        curr_pos_coconuts = position(state, product)
        if curr_pos_coconuts == - 2 * curr_pos_pinacoladas:
            # we are neutral, no hedge orders needed
            return orders

        target_hedge_pos = - 2 * curr_pos_pinacoladas
        order_volume = target_hedge_pos - curr_pos_coconuts

        price = min(state.order_depths[product].sell_orders) \
            if target_hedge_pos > 0 else max(state.order_depths[product].buy_orders)
        orders.append(Order(product, price, order_volume))
        return orders
    

    lastAcceptablePrice_pearls = 10000
    lastAcceptablePrice_bananas = 4800
    lastAcceptablePrice_coconuts = 7000
    lastAcceptablePrice_pina_coladas = 14000
    bananas_max = 20
    pearls_max = 20
    pina_coladas_max = 300
    coconuts_max = 600
    diff_from_mean = 0.005


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """


        # Initialize the method output dict as an empty dict
        result = {}
        # Initialize the list of Orders to be sent as an empty list
        orders: list[Order] = []
        coconuts_data = [8000 for i in range(30)]
        pina_coladas_data = [15000 for i in range(30)]

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():


            if product == 'PEARLS':
                result[product] = self.slack_trader(state=state,
                                                   config=CONFIG_SLACK_TRADER_PEARLS,
                                                   symbol=product,
                                                   price_override=FAIR_PRICE_PEARLS)
            if product == 'BANANAS':
                result[product] = self.slack_trader(state=state,
                                                   config=CONFIG_SLACK_TRADER_BANANAS,
                                                   symbol=product)
                

            if product == 'PEARLS':

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!
                filtered = dict()
                for(key, value) in order_depth.sell_orders.items():
                    if key > 0:
                        filtered[key] = value

                acceptable_price = (self.lastAcceptablePrice_pearls + min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) /3
                self.lastAcceptablePrice_pearls = acceptable_price
                try:
                    pearl_position = state.position[product]
                except:
                    pearl_position = 0

                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value

                    acceptable_ask = (acceptable_price-((self.diff_from_mean/100)*acceptable_price))
                    if (best_ask < acceptable_ask):
                        if ((pearl_position + best_ask_volume) > self.pearls_max):
                            best_ask_volume = ( pearl_position + best_ask_volume)- self.pearls_max
                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY PEARLS", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it finds the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]

                    acceptable_bid = (acceptable_price+((self.diff_from_mean/100)*acceptable_price))
                    if (best_bid > acceptable_bid):
                        #if ((pearl_position - best_bid_volume) < 0):
                            #best_bid_volume = (best_bid_volume + (pearl_position - best_bid_volume))
                        print("SELL PEARLS", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))



            if product == 'BANANAS':

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                
                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!

                acceptable_price = (self.lastAcceptablePrice_bananas + min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) /3
                self.lastAcceptablePrice_bananas = acceptable_price
                try:
                    banana_position = state.position[product]
                except:
                    banana_position = 0
                
                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    acceptable_ask = (acceptable_price-((self.diff_from_mean/100)*acceptable_price))
                    if (best_ask < acceptable_ask):
                        
                        if ((banana_position + best_ask_volume) > self.bananas_max):
                            best_ask_volume = (banana_position + best_ask_volume)- self.bananas_max

                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY BANANAS", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it finds the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    acceptable_bid = (acceptable_price+((self.diff_from_mean/100)*acceptable_price))
                    if (best_bid > acceptable_bid):
                        #if ((banana_position - best_bid_volume) < 0):
                            #best_bid_volume = (best_bid_volume + (banana_position - best_bid_volume))
                        print("SELL BANANAS", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))


            if product == "COCONUTS":
                order_depth_coconuts: OrderDepth = state.order_depths['COCONUTS']
                order_depth_pina_coladas: OrderDepth = state.order_depths['PINA_COLADAS']

                mid_price_coconuts = (min(order_depth_coconuts.sell_orders.keys()) + max(order_depth_coconuts.buy_orders.keys()))/2
                mid_price_pina_coladas = (min(order_depth_pina_coladas.sell_orders.keys()) + max(order_depth_pina_coladas.buy_orders.keys()))/2

                coconuts_data.append(mid_price_coconuts)
                pina_coladas_data.append(mid_price_pina_coladas)

                d = {'coconuts':coconuts_data, 'pina_coladas':pina_coladas_data}
                df = pd.DataFrame(d)

                spread = df['coconuts'] - df['pina_coladas']

                spread_mean = spread.rolling(window=30).mean()
                spread_std = spread.rolling(window=30).std()
                zscore = (spread - spread_mean) / spread_std

                long_signal = zscore  < -2.0
                short_signal = zscore > 2.0
                exit_signal = abs(zscore) < 1.0



                if long_signal[long_signal.size-1]:
                    #long position
                    #buy pina_coladas
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth_pina_coladas.buy_orders.keys())
                        best_ask_volume = order_depth_pina_coladas.buy_orders[best_ask]
                        print("BUY PINA_COLADAS", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order('PINA_COLADAS', best_ask, -best_ask_volume))
                    #sell coconuts
                    if len(order_depth_coconuts.buy_orders) != 0:
                        best_bid = max(order_depth_coconuts.sell_orders.keys())
                        best_bid_volume = order_depth_coconuts.sell_orders[best_bid]
                        print("SELL COCONUTS", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order('COCONUTS', best_bid, -best_bid_volume))

                elif short_signal[short_signal.size-1]:
                    #short position
                    #buy coconuts
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth_coconuts.sell_orders.keys())
                        best_ask_volume = order_depth_coconuts.sell_orders[best_ask]
                        print("BUY COCONUTS", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order('COCONUTS', best_ask, -best_ask_volume))
                    #sell pina_coladas
                    if len(order_depth_pina_coladas.buy_orders) != 0:
                        best_bid = max(order_depth_pina_coladas.buy_orders.keys())
                        best_bid_volume = order_depth_pina_coladas.buy_orders[best_bid]
                        print("SELL PINA_COLADAS", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order('PINA_COLADAS', best_bid, -best_bid_volume))


        # Add all the above orders to the result dict
        result[product] = orders
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        return result
        
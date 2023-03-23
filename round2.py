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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        for symbol in state.listings.keys():

            if symbol == 'PEARLS':
                result[symbol] = self.slack_trader(state=state,
                                                   config=CONFIG_SLACK_TRADER_PEARLS,
                                                   symbol=symbol,
                                                   price_override=FAIR_PRICE_PEARLS)
            if symbol == 'BANANAS':
                result[symbol] = self.slack_trader(state=state,
                                                   config=CONFIG_SLACK_TRADER_BANANAS,
                                                   symbol=symbol)

            if symbol == 'PINA_COLADAS':
                result[symbol] = self.trade_pinacoladas(state=state)

            

        return result
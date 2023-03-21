from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol


class Trader:
    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        result = {}
        spread = 0.01  # Define the spread percentage

        for symbol, order_depth in state.order_depths.items():
            result[symbol] = []
            product = state.listings[symbol].product

            if product == "PEARLS":
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())

                buy_price = best_bid * (1 - spread)
                sell_price = best_ask * (1 + spread)

                # Place buy and sell orders for PEARLS
                result[symbol].append(Order(symbol=symbol, price=buy_price, quantity=1))
                result[symbol].append(Order(symbol=symbol, price=sell_price, quantity=1))

            elif product == "BANANAS":
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())

                buy_price = best_bid * (1 - spread)
                sell_price = best_ask * (1 + spread)

                # Place buy and sell orders for BANANAS
                result[symbol].append(Order(symbol=symbol, price=buy_price, quantity=1))
                result[symbol].append(Order(symbol=symbol, price=sell_price, quantity=1))

        return result

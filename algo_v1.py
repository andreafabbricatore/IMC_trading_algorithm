from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths.keys():
            if product == "PEARLS":
                order_depth: OrderDepth = state.order_depths[product]
                pearl_orders: list[Order] = []

            elif product == "BANANAS":
                # init lists
                order_depth: OrderDepth = state.order_depths[product]
                banana_orders: list[Order] = []

                # check whether to ask or bid first based on volume
                # to be implemented?

                # getting ask, bid prices
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())

                # if best_bid < best_ask go
                price_change = 0
                if best_bid < best_ask:
                    price_change = (best_ask - best_bid) / 4

                our_bid = best_bid + price_change
                our_ask = best_ask - price_change

                banana_orders.append(Order(product, our_bid, 5))
                banana_orders.append(Order(product, our_ask, -5))

                result[product] = banana_orders

        return result

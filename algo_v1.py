from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths.keys():
            if product == "BANANAS":
                order_depth: OrderDepth = state.order_depths[product]
                pearl_orders: list[Order] = []

            elif product == "PEARLS":
                # init lists

                order_depth: OrderDepth = state.order_depths[product]
                pearl_orders: list[Order] = []

                # check whether to ask or bid first based on volume
                # to be implemented?

                # getting ask, bid prices
                try:
                    if state.position[product] == 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_bid = max(order_depth.buy_orders.keys())

                        # if best_bid < best_ask go

                        if best_bid < best_ask:
                            price_change = (best_ask - best_bid) / 8
                            qty = min(
                                abs(order_depth.sell_orders[best_ask]),
                                abs(order_depth.buy_orders[best_bid]),
                            )
                            our_bid = best_bid + price_change
                            our_ask = best_ask - price_change

                            pearl_orders.append(Order(product, our_bid, qty))
                            pearl_orders.append(Order(product, our_ask, -qty))

                            result[product] = pearl_orders
                except:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())

                    # if best_bid < best_ask go

                    if best_bid < best_ask:
                        qty = min(
                            abs(order_depth.sell_orders[best_ask]),
                            abs(order_depth.buy_orders[best_bid]),
                        )
                        price_change = (best_ask - best_bid) / 8
                        our_bid = best_bid + price_change
                        our_ask = best_ask - price_change

                        pearl_orders.append(Order(product, our_bid, qty))
                        pearl_orders.append(Order(product, our_ask, -qty))

                        result[product] = pearl_orders

        return result

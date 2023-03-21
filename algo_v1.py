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

                pearl_orders.append(Order(product, 10001, -40))
                pearl_orders.append(Order(product, 9999, 40))

                result[product] = pearl_orders

        return result

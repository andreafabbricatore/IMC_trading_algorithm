from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths.keys():
            if product == "PEARLS":
                pass
            elif product == "BANANAS":
                # init lists
                order_depth: OrderDepth = state.order_depths[product]
                banana_orderrs: list[Order] = []

                # check whether to ask or bid first based on volume
                # to be implemented?

        return result

from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths.keys():
            if product == "PEARLS":
                pass
            elif product == "BANANAS":
                pass

        return result

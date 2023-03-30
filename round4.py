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

LOT_SIZE = 10
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
        self.pos_limit = {"PEARLS": 20, "BANANAS": 20, "COCONUTS": 600,
                          "PINA_COLADAS": 300, "BERRIES": 250, "DIVING_GEAR": 50, 
                          "BAGUETTE": 150, "DIP":300, "UKULELE": 70, "PICNIC_BASKET":70}
        self.pos = {"PEARLS": 0, "BANANAS": 0, "COCONUTS": 0,
                    "PINA_COLADAS": 0, "BERRIES": 0, "DIVING_GEAR": 0, "BAGUETTE": 0, "DIP":0, "UKULELE": 0, "PICNIC_BASKET":0}
        self.sma = {"PEARLS": [], "BANANAS": [],
                    "BERRIES": [], "DIVING_GEAR": []}
        self.last_timestamp = {"PEARLS": 0, "BANANAS": 0, "COCONUTS": 0,
                               "PINA_COLADAS": 0, "BERRIES": 0, "DIVING_GEAR": 0, "BAGUETTE": 0, "DIP":0, "UKULELE": 0, "PICNIC_BASKET":0}
        self.diffs = []
        self.entered = 0
        self.pc_midprices = []
        self.c_midprices = []
        self.dolphin_sightings = []
        self.combined_midprices = []
        self.basket_midprices = []
        self.dolphin_sightings = []

    def get_order_book_info(self, order_depth):
        best_ask = min(order_depth.sell_orders.keys()) if len(
            order_depth.sell_orders) != 0 else None
        best_ask_volume = order_depth.sell_orders[best_ask] if best_ask is not None else None
        best_bid = max(order_depth.buy_orders.keys()) if len(
            order_depth.buy_orders) != 0 else None
        best_bid_volume = order_depth.buy_orders[best_bid] if best_bid is not None else None
        avg = (best_bid + best_ask) / \
            2 if best_bid is not None and best_ask is not None else None
        return best_ask, best_ask_volume, best_bid, best_bid_volume, avg
    

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


    def trade_berries(self, state: TradingState) -> List[Order]:
        product = "BERRIES"
        limit = 250
        order_depth, position, timestamp = state.order_depths.get(
            product, None), state.position.get(product, 0), state.timestamp
        if not order_depth:
            return []


        # production
        buy_ends_at = 300 * 1000
        sell_starts_at = 500 * 1000

        if timestamp < buy_ends_at and order_depth.sell_orders and position < limit:
            best_ask = min(order_depth.sell_orders.keys())
            volume = order_depth.sell_orders[best_ask]
            return [Order(product, best_ask, -volume)]
        if timestamp > sell_starts_at and order_depth.buy_orders and position > -limit:
            best_bid = max(order_depth.buy_orders.keys())
            volume = order_depth.buy_orders[best_bid]
            return [Order(product, best_bid, -volume)]
        return []

    

    lastAcceptablePrice_pearls = 10000
    lastAcceptablePrice_bananas = 4800
    lastAcceptablePrice_coconuts = 7000
    lastAcceptablePrice_pina_coladas = 14000
    bananas_max = 20
    pearls_max = 20
    pina_coladas_max = 300
    coconuts_max = 600
    diff_from_mean = 0.005

    def pair_trade(self, state: TradingState):
        # Pair trading between COCONUTS and PINA_COLADAS
        order_baguette: list[Order] = []
        order_dip = []
        order_ukulele = []
        order_picnic = []
        order_depth_baguette = state.order_depths["BAGUETTE"]
        order_depth_dip = state.order_depths["DIP"]
        order_depth_ukulele = state.order_depths["UKULELE"]
        order_depth_picnic = state.order_depths["PICNIC_BASKET"]
        best_ask_baguette, best_ask_volume_baguette, best_bid_baguette, best_bid_volume_baguette, avg_baguette = self.get_order_book_info(order_depth_baguette)
        best_ask_dip, best_ask_volume_dip, best_bid_dip, best_bid_volume_dip, avg_dip= self.get_order_book_info(order_depth_dip)
        best_ask_ukulele, best_ask_volume_ukulele, best_bid_ukulele, best_bid_volume_ukulele, avg_ukulele = self.get_order_book_info(order_depth_ukulele)
        best_ask_picnic, best_ask_volume_picnic, best_bid_picnic, best_bid_volume_picnic, avg_picnic = self.get_order_book_info(order_depth_picnic)
        best_ask_combined,best_bid_combined = 15*best_ask_baguette/7 + 30*best_ask_dip/7 + best_ask_ukulele, 15*best_bid_baguette/7 + 30*best_bid_dip/7 + best_bid_ukulele
        avg_combined = (best_ask_combined + best_bid_combined)/2
        # compute normed price difference
        print("entered = ", self.entered)

        if avg_combined is not None and avg_picnic is not None:
            self.combined_midprices.append(avg_combined)
            self.basket_midprices.append(avg_picnic)
            difference = avg_combined - avg_picnic
            self.diffs.append(difference)
            mean = np.array(self.diffs).mean()
            std = np.array(self.diffs).std()
            z = (difference - mean) / std
            print(difference, len(self.diffs), mean, std, z)
            if abs(z) < 0.2:
                print("AAAAAAAAAAAAAAAAAAAAAA")
                self.entered = 0
                product = "BAGUETTE"
                volume = self.pos["BAGUETTE"]
                print("BAGUETTE volume = ", volume)
                if volume > 0:
                    # sell all existing positions
                    print("SELL", product, str(-volume) + "x", best_bid_baguette)
                    order_baguette.append(
                        Order(product, best_bid_baguette, -volume))
                elif volume < 0:
                    # buy all existing positions
                    print("BUY", product, str(-volume) + "x", best_ask_baguette)
                    order_baguette.append(
                        Order(product, best_ask_baguette, -volume))

                product = "DIP"
                volume = self.pos["DIP"]
                if volume > 0:
                    # sell all existing positions
                    print("SELL", product, str(-volume) + "x", best_bid_dip)
                    order_dip.append(Order(product, best_bid_dip, -volume))
                elif volume < 0:
                    # buy all existing positions
                    print("BUY", product, str(-volume) + "x", best_ask_dip)
                    order_dip.append(Order(product, best_ask_dip, -volume))

                product = "UKULELE"
                volume = self.pos["UKULELE"]
                if volume > 0:
                    # sell all existing positions
                    print("SELL", product, str(-volume) + "x", best_bid_ukulele)
                    order_ukulele.append(Order(product, best_bid_ukulele, -volume))
                elif volume < 0:
                    # buy all existing positions
                    print("BUY", product, str(-volume) + "x", best_ask_ukulele)
                    order_ukulele.append(Order(product, best_ask_ukulele, -volume))

                product = "PICNIC_BASKET"
                volume = self.pos["PICNIC_BASKET"]
                if volume > 0:
                    # sell all existing positions
                    print("SELL", product, str(-volume) + "x", best_bid_picnic)
                    order_picnic.append(Order(product, best_bid_picnic, -volume))
                elif volume < 0:
                    # buy all existing positions
                    print("BUY", product, str(-volume) + "x", best_ask_picnic)
                    order_picnic.append(Order(product, best_ask_picnic, -volume))

            elif z > 1:
                print("BBBBBBBBBBBBBBBBBBBBBBBBB")
                # Combined overpriced, basket underpriced
                self.entered += 1
                bid_product = "PICNIC_BASKET"
                ask_product1 = "BAGUETTE"
                ask_product2 = "DIP"
                ask_product3 = "UKULELE"
                bid_volume = min(LOT_SIZE,
                                 self.pos_limit[bid_product] - self.pos[bid_product])
                ask_volume = -min(LOT_SIZE,
                                  int(7/15*(self.pos_limit[ask_product1] + self.pos[ask_product1])),
                                  int(7/30*(self.pos_limit[ask_product2] + self.pos[ask_product2])),
                                  self.pos_limit[ask_product3] + self.pos[ask_product3])
                volume = min(bid_volume, abs(ask_volume))
                bid_volume, ask_volume1, ask_volume2, ask_volume3 = volume, -int(15*volume/7), -int(30*volume/7), -volume
                # TODO: TREAT VOLUME SEPARATELY?
                if bid_volume > 0 and ask_volume < 0:
                    print("BUY", bid_product, str(
                        bid_volume) + "x", best_ask_picnic)
                    order_picnic.append(
                        Order(bid_product, best_ask_picnic, bid_volume))
                    print("SELL", ask_product1, str(
                        ask_volume1) + "x", best_bid_baguette)
                    order_baguette.append(
                        Order(ask_product1, best_bid_baguette, ask_volume1))
                    print("SELL", ask_product2, str(
                        ask_volume2) + "x", best_bid_dip)
                    order_dip.append(
                        Order(ask_product2, best_bid_dip, ask_volume2))
                    print("SELL", ask_product3, str(
                        ask_volume3) + "x", best_bid_ukulele)
                    order_ukulele.append(
                        Order(ask_product3, best_bid_ukulele, ask_volume3))
            elif z < -1:
                # basket overpriced, combined underpriced
                self.entered -= 1
                ask_product = "PICNIC_BASKET"
                bid_product1 = "BAGUETTE"
                bid_product2 = "DIP"
                bid_product3 = "UKULELE"
                bid_volume = min(LOT_SIZE,
                                  int(7/15*(self.pos_limit[bid_product1] - self.pos[bid_product1])),
                                  int(7/30*(self.pos_limit[bid_product2] - self.pos[bid_product2])),
                                  self.pos_limit[bid_product3] -self.pos[bid_product3])
                ask_volume = -min(LOT_SIZE,
                                  self.pos_limit[ask_product] + self.pos[ask_product])
                volume = min(bid_volume, abs(ask_volume))
                ask_volume, bid_volume1, bid_volume2, bid_volume3 = -volume, int(15*volume/7), int(30*volume/7), volume
                # TODO: TREAT VOLUME SEPARATELY?
                print("CCCCCCCCCCCCCCCCCCCCCCCC")
                if bid_volume > 0 and ask_volume < 0:                    
                    print("BUY", bid_product1, str(
                        bid_volume1) + "x", best_ask_baguette)
                    order_baguette.append(
                        Order(bid_product1, best_ask_baguette, bid_volume1))
                    print("BUY", bid_product2, str(
                        bid_volume2) + "x", best_ask_dip)
                    order_dip.append(
                        Order(bid_product2, best_ask_dip, bid_volume2))
                    print("BUY", bid_product3, str(
                        bid_volume3) + "x", best_ask_ukulele)
                    order_ukulele.append(
                        Order(bid_product3, best_ask_ukulele, bid_volume3))               
                    print("SELL", ask_product, str(
                        ask_volume) + "x", best_bid_picnic)
                    order_picnic.append(
                        Order(ask_product, best_bid_picnic, ask_volume))

        return order_ukulele, order_picnic

    def trade_coconut_pinacoladas(self, state: TradingState, product: str) -> List[Order]:
        orders = []

        # Get the related product
        related_product = "COCONUTS" if product == "PINA_COLADAS" else "PINA_COLADAS"

        # get the order depth and position of the products
        c_order_depth: OrderDepth = state.order_depths["COCONUTS"]
        pc_order_depth: OrderDepth = state.order_depths["PINA_COLADAS"]
        
        # Get the best ask and bid prices and volumes for coconut
        c_best_ask = min(c_order_depth.sell_orders.keys())
        c_best_ask_volume = c_order_depth.sell_orders[c_best_ask]
        c_best_bid = max(c_order_depth.buy_orders.keys())
        c_best_bid_volume = c_order_depth.buy_orders[c_best_bid]
        # Get the best ask and bid prices and volumes for pina colada
        pc_best_ask = min(pc_order_depth.sell_orders.keys())
        pc_best_ask_volume = pc_order_depth.sell_orders[pc_best_ask]
        pc_best_bid = max(pc_order_depth.buy_orders.keys())
        pc_best_bid_volume = pc_order_depth.buy_orders[pc_best_bid]

        if pc_best_bid / c_best_ask > 1.876:
            orders.append(Order(product, c_best_ask, -c_best_ask_volume)) if product == "COCONUTS" \
                else orders.append(Order(product, pc_best_bid, -pc_best_bid_volume))
        elif pc_best_ask / c_best_bid < 1.874:
            orders.append(Order(product, c_best_bid, -c_best_bid_volume)) if product == "COCONUTS" \
                else orders.append(Order(product, pc_best_ask, -pc_best_ask_volume))

        return orders

    lastAcceptablePrice = {'PEARLS': 10000, 'BANANAS': 4800, 'COCONUTS':7000, 'PINA_COLADAS':14000, 'DIVING_GEAR': 100000, 'BERRIES':3850 }
    diving_gear_max = 50
    diving_gear_data = []

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """


        # Initialize the method output dict as an empty dict
        result = {}
        for product, trades in state.own_trades.items():
            if len(trades) == 0 or trades[0].timestamp == self.last_timestamp[product]:
                continue
            pos_delta = 0
            for trade in trades:
                print(trade.buyer, trade.seller, trade.price,
                      trade.quantity, trade.symbol)
                if trade.buyer == "SUBMISSION":
                    # We bought product
                    pos_delta += trade.quantity
                    # self.profit -= trade.price * trade.quantity
                elif trade.seller == "SUBMISSION":
                    pos_delta -= trade.quantity
                    # self.profit += trade.price * trade.quantity
            self.pos[product] += pos_delta
            self.last_timestamp[product] = trades[0].timestamp


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
                

           

            order_depth: OrderDepth = state.order_depths[product]

            
            
            if product == "DIVING_GEAR":
                order_depth_diving_gear: OrderDepth = state.order_depths['DIVING_GEAR']

                mid_price_diving_gear = (min(order_depth_diving_gear.sell_orders.keys()) + max(order_depth_diving_gear.buy_orders.keys()))/2
                
                self.diving_gear_data.append(mid_price_diving_gear)
                if len(self.diving_gear_data) > 50:
                    self.diving_gear_data.pop(0)

                d = {'DIVING_GEAR': self.diving_gear_data}
                df = pd.DataFrame(d)

                df['sma'] = df['DIVING_GEAR'].rolling(window=6).mean()


                price = (df['DIVING_GEAR'].tail(1).to_list()[0])
                sma = df['sma'].tail(1).tolist()[0]

                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                avaliable_diving_gear = state.position.get(product)
                if avaliable_diving_gear == None:
                    avaliable_diving_gear = 0

                if (price < sma):
                    orders.append(Order(product, best_ask, -best_ask_volume))
                if (price > sma):
                    orders.append(Order(product, best_bid, -best_bid_volume))

                if avaliable_diving_gear < -self.diving_gear_max*70/100:
                    orders.append(Order(product, mid_price_diving_gear, -best_ask_volume))
                elif avaliable_diving_gear > self.diving_gear_max*70/100:
                    orders.append(Order(product, mid_price_diving_gear, -best_bid_volume))




        # Add all the above orders to the result dict
        for product in state.order_depths.keys():
            filtered = []
            for order in orders:
                if order.symbol == product:
                    if order.symbol != "PEARLS" and order.symbol != "BANANAS":
                        filtered.append(order)
            result[product] = filtered
        
        result["COCONUTS"] = self.trade_coconut_pinacoladas(state, "COCONUTS")
        result["PINA_COLADAS"] = self.trade_coconut_pinacoladas(state, "PINA_COLADAS")
        
        result["UKULELE"],result["PICNIC_BASKET"] = self.pair_trade(state)
        
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        return result
        
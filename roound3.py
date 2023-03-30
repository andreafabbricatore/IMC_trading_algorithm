from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math

LIMITS = {"PEARLS": 20, "BANANAS": 20, "COCONUTS": 600, "PINA_COLADAS": 300}

def dpt(state, product):
    order_depth: OrderDepth = state.order_depths[product]
    cod = state.order_depths["COCONUTS"]
    # Initialize the list of Orders to be sent as an empty list
    orders: list[Order] = []
                
    # Define a fair value for the PEARLS.
    # We will calculate the simple moving average (SMA) of the last N prices
    # and use that as the fair value
    prices = list(order_depth.buy_orders.keys()) + list(order_depth.sell_orders.keys())
    acceptable_price = sum(prices) / len(prices)
    coconut_prices = list(cod.sell_orders.keys()) + list(cod.buy_orders.keys())
    cpi = sum(coconut_prices) / len(coconut_prices)
    if "COCONUTS" in state.market_trades:
        m_trades = state.market_trades["COCONUTS"]
    else:
        m_trades = []
        
    if "COCONUTS" in state.own_trades:
        o_trades = state.own_trades["COCONUTS"]
    else:
        o_trades = []
                
    previous_trades = list(m_trades) + list(o_trades)
    s = 0
    c = 0
    for trade in previous_trades:
        s+=trade.price
        c+=1
    if c:
        pa = s/c
    else:
        pa = cpi
    if pa<cpi:
        if product in state.position:
             pos=state.position[product]
        else:
            pos=0
        bid_quantity = LIMITS[product] - pos
        print("BUY", str(bid_quantity) + "x", acceptable_price, product)
        orders.append(Order(product, math.floor(acceptable_price), bid_quantity))

    if pa>cpi: 
        if product in state.position:
            pos=state.position[product]
        else:
            pos=0
        ask_quantity = -LIMITS[product] - pos
        print("SELL", str(-ask_quantity) + "x", acceptable_price, product)
        orders.append(Order(product, math.ceil(acceptable_price), ask_quantity))
    return orders

def hft(state, product, n):
    
    order_depth: OrderDepth = state.order_depths[product]
    
    orders: list[Order] = []
                
    # Define a fair value for the PEARLS.
    # We will calculate the simple moving average (SMA) of the last N prices
    # and use that as the fair value
    prices = list(order_depth.buy_orders.keys()) + list(order_depth.sell_orders.keys())
    acceptable_price = sum(prices) / len(prices)
                
                
    if product in state.position:
        pos=state.position[product]
    else:
        pos=0
    bid_quantity = LIMITS[product] - pos
    print("BUY", str(bid_quantity) + "x", acceptable_price-n, product)
    orders.append(Order(product, math.floor(acceptable_price)-n, bid_quantity))


    ask_quantity = -LIMITS[product] - pos
    print("SELL", str(-ask_quantity) + "x", acceptable_price+n, product)
    orders.append(Order(product, math.ceil(acceptable_price)+n, ask_quantity))
    
    return orders
    
def lit(state, product, constant):
    # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
    order_depth: OrderDepth = state.order_depths[product]
    # Initialize the list of Orders to be sent as an empty list
    orders: list[Order] = []
                
    # Define a fair value for the PEARLS.
    # We will calculate the simple moving average (SMA) of the last N prices
    # and use that as the fair value
    prices = list(order_depth.buy_orders.keys()) + list(order_depth.sell_orders.keys())
    acceptable_price = sum(prices) / len(prices)
    if product in state.market_trades:
        m_trades = state.market_trades[product]
    else:
        m_trades = []
                
    if product in state.own_trades:
        o_trades = state.own_trades[product]
    else:
        o_trades = []
                
    previous_trades = list(m_trades) + list(o_trades)
    s = 0
    c = 0
    for trade in previous_trades:
        s+=trade.price
        c+=1
    if c:
        pa = s/c
    else:
        pa = acceptable_price
    if pa*constant<acceptable_price:
        if product in state.position:
             pos=state.position[product]
        else:
            pos=0
        bid_quantity = LIMITS[product] - pos
        print("BUY", str(bid_quantity) + "x", acceptable_price, product)
        orders.append(Order(product, math.floor(acceptable_price), bid_quantity))

    if pa>acceptable_price*constant: 
        if product in state.position:
            pos=state.position[product]
        else:
            pos=0
        ask_quantity = -LIMITS[product] - pos
        print("SELL", str(-ask_quantity) + "x", acceptable_price, product)
        orders.append(Order(product, math.ceil(acceptable_price), ask_quantity))
    return orders

def pt(state, product):
    # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
    order_depth: OrderDepth = state.order_depths[product]

    # Initialize the list of Orders to be sent as an empty list
    orders: list[Order] = []
    
    prices = list(order_depth.buy_orders.keys()) + list(order_depth.sell_orders.keys())
    if len(prices):
        acceptable_price = sum(prices) / len(prices)
    else:
        acceptable_price = 10000
    if product in state.position:
        pos=state.position[product]
    else:
        pos=0
    # print("PEARLS: ", acceptable_price)
    if acceptable_price <= 9999.6:
        bid_quantity = LIMITS[product] - pos
        print("BUY", str(bid_quantity) + "x", acceptable_price, product)
        orders.append(Order(product, math.floor(acceptable_price), bid_quantity))
    if acceptable_price >= 10000.4:
        ask_quantity = -LIMITS[product] - pos
        print("SELL", str(-ask_quantity) + "x", acceptable_price, product)
        orders.append(Order(product, math.ceil(acceptable_price), ask_quantity))
    return orders

class Trader:


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        print("<-%")
        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                sell = "sell"
                if trade.buyer == "SUBMISSION":
                    sell = "buy"
                print(trade.symbol, ";", trade.price, ";", trade.quantity, ";", sell, ";", trade.timestamp)
        print("%->")
        result = {}
        if "DOLPHIN_SIGHTINGS" in state.observations:
            print("Sightings: ", state.observations["DOLPHIN_SIGHTINGS"])
        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[product]
            print(order_depth.sell_orders)
            print(order_depth.buy_orders)
            
            # Check if the current product is the 'PEARLS' product, only then run the order logic
            orders: list[Order] = []
            if product == "PEARLS":
                orders = hft(state, product, 1)
            elif product == "BANANAS":
                orders = hft(state, product, 1)
            elif product == "PINA_COLADAS":
                orders = dpt(state, product)
            elif product == "COCONUTS":
                orders = lit(state, product, 1.001)

            result[product] = orders

        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        
        return result
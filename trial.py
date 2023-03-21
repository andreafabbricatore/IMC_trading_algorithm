from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


# Constants
# MA_100POWER_BANANA = 40 / 100
# EMA_100POWER_BANANA = 60 / 100
# MA_POWER_PEARLS = 65 / 100
# EMA_POWER_PEARLS = 35 / 100
# MA_POWER = 40 / 100
# EMA_POWER = 60 / 100


pearls_q = []
bananas_q = []

EMA_yesterday_bananas = 0
EMA_yesterday_pearls = 0

highestPearls_value = -999999
lowestPearls_value = 999999

highestBananas_value = -999999
lowestBananas_value = 999999

previousPearls_Sell = 0
previosPearls_Buy = 0

SoltTotal = 0

Pearls_Owned = 0


def ema_calc(close_today, n):
    global EMA_yesterday_bananas
    EMA_today = (close_today * (2 / (n + 1))) + (
        EMA_yesterday_bananas * (1 - (2 / (n + 1)))
    )
    EMA_yesterday_bananas = EMA_today
    return EMA_today


trend_calculator_q = []
trend_calculator_algo = []

tendayCompute_Trend = 0

AllTimesAverage_q = []

sumAllTime = 0
impart = 0


def trend_calculator(average, allTimeAverage):
    global trend_calculator_q
    global trend_calculator_algo
    global tendayCompute_Trend
    tendayCompute_Trend += 1
    sume = 0
    if len(trend_calculator_q) < 10:
        trend_calculator_q.append(average)
    else:
        trend_calculator_q.pop()
    for el in trend_calculator_q:
        sume += el

    avg = sume / len(trend_calculator_q)

    if len(trend_calculator_algo) < 10:
        trend_calculator_algo.append(avg)
    else:
        trend_calculator_algo.pop()

    avgAns = avg

    return allTimeAverage - avgAns


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product == "PEARLS":
                global SoltTotal
                global EMA_yesterday_pearls
                global highestPearls_value
                global lowestPearls_value

                global maxPearls
                global maxProfitPearls

                global Pearls_Owned

                percent = 1.5 / 10000

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []
                best_bid2 = max(order_depth.buy_orders.keys())
                best_bid_volume2 = order_depth.buy_orders[best_bid2]
                best_ask2 = min(order_depth.sell_orders.keys())
                best_ask_volume2 = order_depth.sell_orders[best_ask2]

                highestPearls_value = max(highestPearls_value, best_bid2)
                lowestPearls_value = min(lowestPearls_value, best_ask2)

                profit = best_bid2 - best_ask2

                mid_price = (best_ask2 + best_bid2) / 2
                print("midPrice: ", mid_price)
                pearls_q.append(mid_price)

                if len(pearls_q) > 100:
                    pearls_q.pop()

                average = 0
                for val in pearls_q:
                    average += val

                average /= len(pearls_q)

                Close_today = mid_price

                n = 25

                EMA_today = (Close_today * (2 / (n + 1))) + (
                    EMA_yesterday_pearls * (1 - (2 / (n + 1)))
                )
                EMA_yesterday_pearls = EMA_today

                average_ema = EMA_today

                if len(pearls_q) < 1000:
                    acceptable_price = average
                else:
                    acceptable_price = average_ema

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!

                print("average is: ", acceptable_price)

                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    bestAsks = []
                    for key, value in order_depth.sell_orders.items():
                        if key < average:
                            print("KEY BAGATA: ", key, "VALUE BAGATA: ", value)
                            bestAsks.append((key, value))

                    print("This is the lowest ask : ")
                    print(best_ask)

                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    # acceptable_price
                    if best_ask < int(average):
                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        # print("BUY", str(acceptable_price - best_ask) + "x", best_ask)
                        # print("BUY", best_ask_volume,"x", best_ask)
                        for key, value in bestAsks:
                            print("KEY ", key, "VALUE ", value)

                        for key, value in bestAsks:
                            orders.append(Order(product, key, -value))
                        #
                        # orders.append(Order(product, best_ask, -best_ask_volume))
                        SoltTotal -= best_ask * -best_ask_volume
                        # orders.append(Order(product,lowestPearls_value,-best_ask_volume))
                        # orders.append(Order(product, best_ask, -best_ask_volume + 1))
                        # orders.append(Order(product, best_ask, -best_ask_volume - 1))
                        # orders.append(Order(product, best_ask-1, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium

                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]

                    bestAsks2 = []
                    for key, value in order_depth.buy_orders.items():
                        if key > average:
                            bestAsks2.append((key, value))

                    if best_bid > int(average):
                        # print("SELL", str(acceptable_price - best_bid) + "x", best_bid)
                        print("SELL", best_bid, "Biggest Bid")
                        for key, value in bestAsks2:
                            orders.append(Order(product, key, -value))
                        SoltTotal += best_bid * best_bid_volume
                        print("SOLD: ", SoltTotal)
                        # orders.append(Order(product, best_bid,-best_bid_volume))
                        # orders.append(Order(product,highestPearls_value,-best_bid_volume))

                        # orders.append(Order(product, best_bid, -best_bid_volume))
                        # orders.append(Order(product, best_bid, -best_bid_volume + 1))
                        # orders.append(Order(product, best_bid, -best_bid_volume - 1))
                        # orders.append(Order(product, best_bid+1 ,-best_bid_volume))

                # Add all the above the orders to the result dict

                result[product] = orders

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product == "BANANAS":

                global EMA_yesterday_bananas
                global AllTimesAverage_q
                global sumAllTime
                global impart

                # global highestBananas_value
                # global lowestBananas_value

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                best_bid2 = max(order_depth.buy_orders.keys())
                best_ask2 = min(order_depth.sell_orders.keys())

                mid_price = (best_ask2 + best_bid2) / 2

                impart += 1
                sumAllTime += mid_price
                AverageAllTime = sumAllTime / impart

                SOS = trend_calculator(mid_price, AverageAllTime)
                letsSee = trend_calculator(mid_price, AverageAllTime)
                print("TREND CALCULATOR : ", letsSee)

                # print("midPrice: ", mid_price)
                bananas_q.append(mid_price)
                if EMA_yesterday_bananas == 0:
                    EMA_yesterday_bananas = mid_price

                if len(bananas_q) > 100:
                    bananas_q.pop()

                average = 0
                for val in bananas_q:
                    average += val

                average /= len(bananas_q)

                Close_today = mid_price

                average_ema = ema_calc(Close_today, 25)

                if len(bananas_q) < 15:
                    acceptable_price = average
                else:
                    acceptable_price = average_ema

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!

                # print("average is: ", acceptable_price)

                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    bestAsks2 = []
                    # asta inca nu i folosita
                    for key, value in order_depth.buy_orders.items():
                        if key < acceptable_price:
                            bestAsks2.append((key, value))

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < acceptable_price:
                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY", str(acceptable_price - best_ask) + "x", best_ask)
                        for key, value in bestAsks2:
                            orders.append(Order(product, key, -value))
                        # orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium

                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]

                    # asta inca nu i folosita
                    bestAsks = []
                    for key, value in order_depth.buy_orders.items():
                        if key > acceptable_price:
                            bestAsks.append((key, value))

                    if best_bid > acceptable_price:
                        print("SELL", str(acceptable_price - best_bid) + "x", best_bid)
                        print("Average All Time ", AverageAllTime)
                        acCompute = trend_calculator(mid_price, AverageAllTime)
                        print("Algo Compute", acCompute)
                        for key, value in bestAsks:
                            orders.append(Order(product, key, -value))
                        # orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above the orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        return result

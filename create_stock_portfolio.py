import requests
from collections import defaultdict
import datetime
import json
from dateutil import parser


def get_Tip_data(symbol, session):
    milliseconds = int(datetime.datetime.now().timestamp() * 1000)
    sym = symbol[0].strip('') if isinstance(symbol, list) else symbol.strip('')
    # res = requests.get("https://www.tipranks.com/stocks/spwr/stock-analysis")
    tip_data = defaultdict(dict)
    try:
        res = session.get(
            "https://www.tipranks.com/api/stocks/getData/?name={}&benchmark=1&period=3&break={}".format(sym,
                                                                                                        milliseconds))
        res_ticker = session.get(
            "https://www.tipranks.com/api/stocks/getChartPageData/?ticker={}&benchmark=1&period=3&break={}".format(sym,
                                                                                                                   milliseconds))
        res.raise_for_status()
        date_dicts = defaultdict(dict)
        s = res.json()
        # Sort by dates
        for data in res.json()['consensusOverTime']:
            date_dicts[data['date']] = data

        # Add to Tipranks stock data
        tip_data['consensusOverTime'] = date_dicts
        tip_data['bloggerSentiment'] = res.json()['bloggerSentiment']
        tip_data['similarStocks'] = res.json()['similarStocks']
        tip_data['topStocksBySector'] = res.json()['topStocksBySector']
        tip_data['tipranksStockScore'] = res.json()['tipranksStockScore']
        tip_data['ticker_data'] = res_ticker.json() if res_ticker else []
        return tip_data
    except requests.exceptions.HTTPError:
        return tip_data


def get_6M_price(stocks, days_ago, session):
    res = session.post(url='https://www.marketbeat.com/Pages/CompareStocks.aspx/GetChartData',
                       json={'stocks': [stocks[0]], 'lookback': days_ago})
    s = res.json()
    date_dicts = defaultdict(dict)
    # Sort by dates
    for data in res.json()['d']['StockRows']:
        date_dicts[data['ItemDate']][data['Symbol']] = data
    return date_dicts


def get_stock_price_SA(symbol, session):
    all_stock_dict = defaultdict(dict)
    periods = ['1D', '5D', '1M', '6M']
    try:
        # Gets Stock's Company Name
        res = session.get(
            'https://finance.api.seekingalpha.com/v2/real-time-prices?symbols%5B%5D={}'.format(symbol)).json()
        name = res['data'][0]['attributes']['name']
        all_stock_dict['name'] = name
        # Adds stock price data for each period declared
        all_stock_dict['real_time'][datetime.datetime.today().strftime("%m/%d/%Y")] = res['data'][0]['attributes'][
            'last']
        for period in periods:
            period_dict = defaultdict(dict)
            temp_dict = defaultdict(dict)
            res_chart = session.get(
                'https://finance.api.seekingalpha.com/v2/chart?period={}&symbol={}&interval=0'.format(period, symbol))
            res_chart.raise_for_status()
            # Content is in the request, as a "byte" format
            string = res_chart.content.decode('utf-8')
            temp_dict = json.loads(string)
            # Sort by dates
            for data in temp_dict['attributes']:
                period_dict[data] = temp_dict['attributes'].get(data)
            all_stock_dict[period] = period_dict
        return all_stock_dict

    except requests.exceptions.HTTPError:
        print("Error getting Seeking Alpha data - {} stock".format(symbol))
        return all_stock_dict


def rank_Tip_accuracy(portfolio):
    for stock in portfolio:
        accuracy_distribution = []
        count_success = 0
        if portfolio[stock]['Tiprank'].get('consensusOverTime', None):
            ana_list = sorted(list(portfolio[stock]['Tiprank']['consensusOverTime'].keys()))
            for date in ana_list:
                parsed_date = parser.parse(date)
                i = 7
                week_price_list = [0]
                bound_date = parser.parse(ana_list[-1]) - datetime.timedelta(days=168)
                if parsed_date > bound_date:
                    while i >= 0:
                        try:
                            delta_date = parsed_date - datetime.timedelta(days=i)
                            i -= 1
                            delta_date_format = delta_date.strftime("%Y-%m-%d")
                            price_usd = portfolio[stock]['stock_data']['6M_Marketbeat'][delta_date_format][stock][
                                'ClosingPrice']
                            week_price_list.append(price_usd)
                        except:
                            i -= 1
                max_price = max(week_price_list)
                parsed_date_format = parsed_date.strftime("%Y-%m-%d")
                priceTarget = portfolio[stock]['Tiprank']['consensusOverTime'][date]['priceTarget']
                if priceTarget and max_price != 0:
                    accuracy_distribution.append(priceTarget - max_price)
                    if priceTarget - max_price < max_price * 0.05:
                        count_success += 1
                else:
                    continue
            if len(accuracy_distribution) > 0:
                portfolio[stock]['Tiprank']['Tip_accuracy'] = (
                    count_success / len(accuracy_distribution) * 100,
                    '# of weeks:{}'.format(len(accuracy_distribution)))


# Extracts similar and top stocks by sector and adds to each stock in portfolio
def add_similar_stocks(stock_data, stock, spdr_etfs, source_field_name, target_field_name):
    similar_stocks = defaultdict(dict)
    if stock_data['Tiprank'].get(source_field_name, None):
        # Adds similar stocks
        for iter in stock_data['Tiprank'][source_field_name]:
            if source_field_name != 'topStocksBySector':
                symbol = iter['ticker']
                similar_stocks[symbol] = symbol if symbol not in similar_stocks.keys() else None
            else:
                for recommenders in stock_data['Tiprank'][source_field_name][iter]:
                    symbol = recommenders['ticker']
                    similar_stocks[symbol] = symbol if symbol not in similar_stocks.keys() else None
    # Handles ETFs not in Tipranks data -
    if stock in spdr_etfs:
        for etf in spdr_etfs:
            similar_stocks[etf] = etf if etf not in similar_stocks.keys() else None
    stock_data[target_field_name] = similar_stocks


def create_Portfolio(stocks):
    # Default symbols for SPDR ETF's
    spdr_etfs = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
    # Default for Marketbeat.com - in order to get 6 months stock history
    days_ago = 180
    portfolio = defaultdict(dict)
    session = requests.session()
    has_amount = isinstance(stocks[0], list)
    # Deals when it only gets a list of stock symbols (not in format of [stock symbol, stock amount])
    if has_amount:
        stocks_symbols = [x[0] for x in stocks]
    else:
        stocks_symbols = stocks
    for stock in stocks_symbols:
        stock_data = defaultdict(dict)
        # adds Tiprank stock data
        stock_data['Tiprank'] = get_Tip_data(stock, session)
        add_similar_stocks(stock_data, stock, spdr_etfs, 'similarStocks', 'similar_stocks')# creates similar stocks data
        add_similar_stocks(stock_data, stock, spdr_etfs, 'topStocksBySector', 'top_stocks')# creates top stocks by sector data
        # Adds stock prices
        stock_data['stock_data'] = get_stock_price_SA(stock, session) #Seeking Alpha
        stock_data['stock_data']['6M_Marketbeat'] = get_6M_price([stock], days_ago, session)# Marketbeat
        # Appends stock data to Portfolio
        portfolio[stock] = stock_data
        # Adds amounts for each stock (if exists)
        if has_amount:
            portfolio[stock]['amount'] = [t[1] for t in stocks if t[0] == stock][0]
        else:
            continue
    # Adds Tiprank analyst accuracy over the past 6 months
    rank_Tip_accuracy(portfolio)
    return portfolio

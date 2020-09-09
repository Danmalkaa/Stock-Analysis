import requests
from collections import defaultdict
import datetime
import json
from dateutil import parser
import math
import calc_stock_hour_correlation


def get_tip_data(symbol, session):
    ms = int(datetime.datetime.now().timestamp() * 1000)
    sym = symbol[0].strip('') if isinstance(symbol, list) else symbol.strip('')
    # res = requests.get("https://www.tipranks.com/stocks/spwr/stock-analysis")
    tip_data = defaultdict(dict)
    try:
        res = session.get(
            "https://www.tipranks.com/api/stocks/getData/?name={}&benchmark=1&period=3&break={}".format(sym, ms))
        res_ticker = session.get(
            "https://www.tipranks.com/api/stocks/getChartPageData/?ticker={}&benchmark=1&period=3&break={}".format(sym,
                                                                                                                   ms))
        res.raise_for_status()
        date_dicts = defaultdict(dict)
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


def get_6m_price(stocks, days_ago, session):
    res = session.post(url='https://www.marketbeat.com/Pages/CompareStocks.aspx/GetChartData',
                       json={'stocks': [stocks[0]], 'lookback': days_ago})
    date_dicts = defaultdict(dict)
    # Sort by dates
    for data in res.json()['d']['StockRows']:
        date_dicts[data['ItemDate']][data['Symbol']] = data
    return date_dicts


def get_stock_price_sa(symbol, session):
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


def rank_tip_accuracy(portfolio):
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
                        delta_date = parsed_date - datetime.timedelta(days=i)
                        i -= 1
                        delta_date_format = delta_date.strftime("%Y-%m-%d")
                        exists = portfolio[stock]['stock_data']['6M_Marketbeat'].get(delta_date_format, None)
                        if exists:
                            price_usd = portfolio[stock]['stock_data']['6M_Marketbeat'][delta_date_format][stock][
                                'ClosingPrice']
                            week_price_list.append(price_usd)
                        else:
                            i -= 1
                max_price = max(week_price_list)
                # parsed_date_format = parsed_date.strftime("%Y-%m-%d")
                pricetarget = portfolio[stock]['Tiprank']['consensusOverTime'][date]['priceTarget']
                if pricetarget and max_price != 0:
                    accuracy_distribution.append(pricetarget - max_price)
                    if pricetarget - max_price < max_price * 0.05:
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
        for sim in stock_data['Tiprank'][source_field_name]:
            if source_field_name != 'topStocksBySector':
                symbol = sim['ticker']
                similar_stocks[symbol] = symbol if symbol not in similar_stocks.keys() else None
            else:
                for recommenders in stock_data['Tiprank'][source_field_name][sim]:
                    symbol = recommenders['ticker']
                    similar_stocks[symbol] = symbol if symbol not in similar_stocks.keys() else None
    # Handles ETFs not in Tipranks data -
    if stock in spdr_etfs:
        for etf in spdr_etfs:
            similar_stocks[etf] = etf if etf not in similar_stocks.keys() else None
    stock_data[target_field_name] = similar_stocks


def create_portfolio(stocks):
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
        stock_data['Tiprank'] = get_tip_data(stock, session)
        add_similar_stocks(stock_data, stock, spdr_etfs, 'similarStocks',
                           'similar_stocks')  # creates similar stocks data
        add_similar_stocks(stock_data, stock, spdr_etfs, 'topStocksBySector',
                           'top_stocks')  # creates top stocks by sector data
        # Adds stock prices
        stock_data['stock_data'] = get_stock_price_sa(stock, session)  # Seeking Alpha
        stock_data['stock_data']['6M_Marketbeat'] = get_6m_price([stock], days_ago, session)  # Marketbeat
        # Appends stock data to Portfolio
        portfolio[stock] = stock_data
        # Adds amounts for each stock (if exists)
        if has_amount:
            portfolio[stock]['amount'] = [t[1] for t in stocks if t[0] == stock][0]
        else:
            continue
    # Adds Tiprank analyst accuracy over the past 6 months
    rank_tip_accuracy(portfolio)
    return portfolio


def check_difference(stock, rival_portfolio, start_date, end_date=datetime.datetime.today()):  # yield chg percentage
    try:
        if isinstance(end_date, datetime.datetime):
            price_usd_target = rival_portfolio[stock]['stock_data']['real_time'][
                list(rival_portfolio[stock]['stock_data']['real_time'].keys())[0]]
        else:
            date = datetime.datetime.strptime(end_date, '%Y-%m-%d.%f')
            price_usd_target = rival_portfolio[stock]['stock_data']['6M_Marketbeat'][date]
        price_usd_start = rival_portfolio[stock]['stock_data']['6M_Marketbeat'][start_date][stock]['ClosingPrice']

        chg = price_usd_target / price_usd_start * 100 - 100
        return chg
    except KeyError:
        chg = -1
        return chg


def calc_last_stock_price(price_usd_target, target_date, portfolio, stock):
    price_flag = False
    if price_usd_target is None:
        j = 0
        while j <= 7 and price_usd_target is None:
            formated_target_date = datetime.datetime.strptime(target_date, '%Y-%M-%d')
            edate = formated_target_date - datetime.timedelta(days=j)
            date = edate.strftime("%Y-%m-%d")
            stock_data = portfolio[stock]['stock_data']['6M_Marketbeat'].get(date, None)
            price_usd_target = stock_data[list(stock_data.keys())[0]]['ClosingPrice'] if stock_data else None
            j += 1
            price_flag = date if price_usd_target else None
            if price_flag:
                portfolio[stock]['stock_data']['date_for_calculation'] = price_flag
    if price_flag:
        return price_usd_target
    else:
        print("Problem with {} stock price date {} and 7 days before".format(stock, target_date))
        return 0


def calc_equiv_stock_amount(total_usd_start, stock, portfolio, startdate_format):
    date_exists = portfolio[stock]['stock_data']['6M_Marketbeat'].get(startdate_format, None)
    price_usd_start = portfolio[stock]['stock_data']['6M_Marketbeat'][startdate_format][stock][
        'ClosingPrice'] if date_exists else None
    if price_usd_start is None:
        price_usd_start = calc_last_stock_price(price_usd_start, startdate_format, portfolio, stock)
    if price_usd_start == 0:
        amount = -1
    else:
        amount = math.floor(total_usd_start / price_usd_start)
    return amount


def calc_performance(portfolio, test, period, end_date=datetime.datetime.today(),
                     with_rival=True):  # yield chg percentage
    if test:
        # Loads portfolio from file - Test purpose for dev
        f = open("22.4_RIVAL.json", "r")
        rival_portfolio = json.load(f)
        for s in rival_portfolio:
            calc_stock_hour_correlation.stock_hour_correlation(rival_portfolio, s)
    # Creates Similar Stocks list and Portfolio
    # with_rival is True - will create another portfolio for similar stocks
    if with_rival and not test:
        rival_stocks = defaultdict(dict)
        for stock in portfolio.keys():
            similar = portfolio[stock]['similar_stocks']
            for t in portfolio[stock]['top_stocks'].keys():
                similar[t] = portfolio[stock]['top_stocks'][t]
            if len(similar) > 0:
                for st in similar:
                    rival_stocks[st] = st
            else:
                continue
        rival_list = list(rival_stocks.keys())
        rival_portfolio = create_portfolio(rival_list)
        for s in rival_portfolio:
            calc_stock_hour_correlation.stock_hour_correlation(rival_portfolio, s)
        # Save Portfolio to file
        json_r = json.dumps(rival_portfolio)
        f = open("22.4_RIVAL.json", "w")
        f.write(json_r)
        f.close()
    periods = [7, 14, 30]  # Default Periods
    # Adds period called to the default periods
    periods.append(period) if period not in periods else None
    # Calc performance for my portfolio
    chg = defaultdict(dict)
    lacking_data = []
    price_usd_start = 0.0
    try:
        for per in periods:
            currentday = end_date if isinstance(end_date, datetime.datetime) else datetime.datetime.strptime(end_date,
                                                                                                             '%Y-%m-%d')
            startdate = currentday - datetime.timedelta(days=per)
            sum_target, sum_start = 0, 0
            is_better_dict = defaultdict(dict)
            startdate_format = startdate.strftime("%Y-%m-%d")
            for stock in portfolio:
                # Adds Correlations to stock data in portfolio
                calc_stock_hour_correlation.stock_hour_correlation(portfolio, stock)
                # Calcs price performance
                amount = portfolio[stock]['amount']
                if end_date is not calc_performance.__defaults__[0]:
                    date = datetime.datetime.today().strftime("%Y-%m-%d")
                    price_usd_target = portfolio[stock]['stock_data']['6M_Marketbeat'][date]
                    total_usd_target = amount * price_usd_target
                else:
                    price_usd_target = portfolio[stock]['stock_data']['real_time'][
                        list(portfolio[stock]['stock_data']['real_time'].keys())[0]]
                    # Handles stocks with no real time data
                    if price_usd_target is None:
                        calc_last_stock_price(price_usd_target, currentday, portfolio, stock)
                    total_usd_target = amount * price_usd_target
                price_usd_start = calc_last_stock_price(price_usd_start, startdate_format, portfolio, stock)
                # price_usd_start = portfolio[stock]['stock_data']['6M_Marketbeat'][startdate_format][list(portfolio[stock]['stock_data']['6M_Marketbeat'][startdate_format].keys())[0]]['ClosingPrice']
                total_usd_start = amount * price_usd_start
                stock_chg = total_usd_target / total_usd_start * 100 - 100 if not total_usd_start == 0 else 0
                yields = defaultdict(dict)
                yields['chg last {} days'.format(per)] = stock_chg
                portfolio[stock]['stock_data']['yield'] = yields
                if with_rival:
                    # Check whether the similar stocks performed better
                    better_list = []
                    if portfolio[stock].get('similar_stocks', None):
                        for rstock in portfolio[stock]['similar_stocks']:
                            rival_chg = check_difference(rstock, rival_portfolio, startdate_format, end_date)
                            if rival_chg == -1:
                                lacking_data.append(rstock) if rstock not in lacking_data else None
                            if rival_chg > stock_chg:
                                s_amount = calc_equiv_stock_amount(total_usd_start, rstock, rival_portfolio,
                                                                   startdate_format)
                                if s_amount == -1:  # If Error in fetching stocks prices
                                    continue
                                new_tripl = [rstock, s_amount, rival_chg]
                                better_list.append(new_tripl)
                            else:
                                continue
                    is_better_dict[(stock, stock_chg)] = better_list

                # Sums up the amounts for the current portfolio performance
                sum_target += total_usd_target
                sum_start += total_usd_start
                chg['{} days'.format(per)] = 100 * (sum_target / sum_start) - 100 if sum_start else None
        if lacking_data:
            print("No data for {} stocks".format(lacking_data))
        if with_rival:
            return chg, is_better_dict, rival_portfolio
        else:
            return chg
    except TypeError:
        print("Error creating portfolio for period {} days".format(per))


def create_alternative_portfolio(better_dict, portfolio, alt_stock_data):
    new_portfolio = dict(portfolio)
    new_list_stocks = []
    i = 0
    for stock in better_dict:
        if better_dict[stock]:
            new_portfolio.pop(stock[0])

            def func(p):
                return p[2]

            best = max(better_dict[stock], key=func, default=[0, 0, 0])
            new_list_stocks.append(best[:2])
            best_stock_name = new_list_stocks[i][0]
            alt_stock_data[best_stock_name]['amount'] = new_list_stocks[i][1]
            new_portfolio[best_stock_name] = alt_stock_data[best_stock_name]
            i += 1
    return new_portfolio

from typing import List, Any
#import tkinter
import requests
from collections import defaultdict
import datetime
import json
import math
import re
import numpy
from sklearn import preprocessing
import pandas as pd
import statistics
from dateutil import parser
from scipy.stats import pearsonr
from operator import or_, __xor__
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.optimize as opt

def get_Tip_data(symbol, session):
    milliseconds =int(datetime.datetime.now().timestamp() * 1000)
    sym = symbol[0].strip('') if isinstance(symbol, list) else symbol.strip('')
    #res = requests.get("https://www.tipranks.com/stocks/spwr/stock-analysis")
    tip_data = defaultdict(dict)
    try:
        res = session.get("https://www.tipranks.com/api/stocks/getData/?name={}&benchmark=1&period=3&break={}".format(sym,milliseconds))
        res_ticker = session.get("https://www.tipranks.com/api/stocks/getChartPageData/?ticker={}&benchmark=1&period=3&break={}".format(sym,milliseconds))
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
        tip_data['ticker_data'] = res_ticker.json()
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

def get_stock_price_SA(symbol,session):

    all_stock_dict = defaultdict(dict)
    periods = ['1D','5D','1M','6M']
    try:
        # Gets Stock's Company Name
        res = session.get('https://finance.api.seekingalpha.com/v2/real-time-prices?symbols%5B%5D={}'.format(symbol)).json()
        name = res['data'][0]['attributes']['name']
        all_stock_dict['name'] = name
        # Adds for stock price data for each period declared
        all_stock_dict['real_time'][datetime.datetime.today().strftime("%m/%d/%Y")] = res['data'][0]['attributes']['last']
        for period in periods:
            period_dict = defaultdict(dict)
            temp_dict = defaultdict(dict)
            res_chart = session.get('https://finance.api.seekingalpha.com/v2/chart?period={}&symbol={}&interval=0'.format(period,symbol))
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
        if portfolio[stock]['Tiprank'].get('consensusOverTime',None):
            ana_list=sorted(list(portfolio[stock]['Tiprank']['consensusOverTime'].keys()))
            for date in ana_list:
                parsed_date = parser.parse(date)
                i = 7
                week_price_list = [0]
                bound_date = parser.parse(ana_list[-1]) - datetime.timedelta(days = 168)
                if parsed_date > bound_date:
                    while i >= 0 :
                        try:
                            delta_date = parsed_date - datetime.timedelta(days=i)
                            i -= 1
                            delta_date_format = delta_date.strftime("%Y-%m-%d")
                            price_usd = portfolio[stock]['stock_data']['6M_Marketbeat'][delta_date_format][stock]['ClosingPrice']
                            week_price_list.append(price_usd)
                        except:
                            i -= 1
                max_price = max(week_price_list)
                parsed_date_format = parsed_date.strftime("%Y-%m-%d")
                priceTarget = portfolio[stock]['Tiprank']['consensusOverTime'][date]['priceTarget']
                if priceTarget and max_price!= 0:
                    accuracy_distribution.append(priceTarget - max_price)
                    if priceTarget-max_price < max_price*0.05 :
                        count_success += 1
                else: continue
            if len(accuracy_distribution)>0 :
                portfolio[stock]['Tiprank']['Tip_accuracy'] = (count_success/len(accuracy_distribution)*100,'# of weeks:{}'.format(len(accuracy_distribution)))

def create_Portfolio(stocks):
    spdr_etfs = ['XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLU']
    days_ago = 180
    portfolio = defaultdict(dict)
    session = requests.session()
    has_amount = isinstance(stocks[0], list)
    if has_amount:
        stocks_symbols = [x[0] for x in stocks]
    else : stocks_symbols = stocks
    for stock in stocks_symbols:
        stock_data = defaultdict(dict)
        similar_stocks = defaultdict(dict)
        #adds Tiprank stock data
        stock_data['Tiprank'] = get_Tip_data(stock, session)
        #Adds similar stocks
        for iter in stock_data['Tiprank']['similarStocks']:
            symbol = iter['ticker']
            similar_stocks[symbol] = symbol if symbol not in similar_stocks.keys() else None
        if stock in spdr_etfs :
            for etf in spdr_etfs:
                similar_stocks[etf] = etf if etf not in similar_stocks.keys() else None
        stock_data['similar_stocks'] = similar_stocks
        #Adds stock prices
        stock_data['stock_data'] = get_stock_price_SA(stock, session)
        stock_data['stock_data']['6M_Marketbeat'] = get_6M_price([stock], days_ago, session)
        #Appends stock data to Portfolio
        portfolio[stock] = stock_data
        if  has_amount:
            portfolio[stock]['amount'] = [t[1] for t in stocks if t[0] == stock][0]
        else: continue
    #Adds Tiprank Accuracy
    rank_Tip_accuracy(portfolio)
    return portfolio

def check_difference(stock, rival_portfolio, start_date, end_date=datetime.datetime.today()):  ## yeild chg percentage
    chg = 0.0
    try:
        if isinstance(end_date, datetime.datetime):
            price_usd_target = rival_portfolio[stock]['stock_data']['real_time'][
                list(rival_portfolio[stock]['stock_data']['real_time'].keys())[0]]
        else:
            date = end_date.datetime.datetime.strptime(end_date, '%Y-%m-%d.%f')
            price_usd_target = rival_portfolio[stock]['stock_data']['6M_Marketbeat'][date]
        price_usd_start = rival_portfolio[stock]['stock_data']['6M_Marketbeat'][start_date][stock]['ClosingPrice']

        chg = price_usd_target / price_usd_start * 100 - 100
        return chg
    except :
        print("No data for {} stock".format(stock))
        return chg

def calc_equiv_stock_amount(total_usd_start,stock,portfolio,startdate_format):
    price_usd_start = 0.0
    price_usd_start = portfolio[stock]['stock_data']['6M_Marketbeat'][startdate_format][stock]['ClosingPrice']
    amount = math.floor(total_usd_start/price_usd_start)
    return amount

def calc_performance(portfolio,test,period,end_date=datetime.datetime.today(), withRival = True):  ##yeild chg percentage
    if test:
        ##Loads portfolio from file - Test purpose
        f = open("22.4_RIVAL.json", "r")
        rival_portfolio = json.load(f)
    # Creates Similar Stocks list and Portfolio
    if withRival and not test:
        rival_stocks = defaultdict(dict)
        for stock in portfolio.keys():
            similar = portfolio[stock]['similar_stocks']
            if len(similar) > 0:
                for st in similar:
                    rival_stocks[st] = st
            else:
                continue
        rival_list = list(rival_stocks.keys())
        rival_portfolio = create_Portfolio(rival_list)
        ##Save Portfolio to file
        json_r = json.dumps(rival_portfolio)
        f = open("22.4_RIVAL.json", "w")
        f.write(json_r)
        f.close()
    # Calc performance for my portfolio
    periods = [5,7,30]#,14]
    periods.append(period) if period not in periods else None
    chg = defaultdict(dict)
    for per in periods:
        rival_chg= 0.0
        currentday = end_date if isinstance(end_date,datetime.datetime) else end_date.datetime.datetime.strptime(end_date, '%Y-%m-%d.%f')
        startdate = currentday - datetime.timedelta(days=per)
        sum_target, sum_start = 0, 0
        date = end_date
        is_better_dict = defaultdict(dict)
        startdate_format = startdate.strftime("%Y-%m-%d")
        for stock in portfolio:
            amount = portfolio[stock]['amount']
            if end_date is not calc_performance.__defaults__[0]:
                date = end_date.datetime.datetime.today().strftime("%Y-%m-%d")
                price_usd_target = portfolio[stock]['stock_data']['6M_Marketbeat'][date]
                total_usd_target = amount * price_usd_target
            else:
                price_usd_target = portfolio[stock]['stock_data']['real_time'][list(portfolio[stock]['stock_data']['real_time'].keys())[0]]
                #Handles stocks with no real time data
                if (price_usd_target==None) :
                    j=0
                    while j<=7 and price_usd_target == None:
                        edate = currentday - datetime.timedelta(days=j)
                        date = edate.strftime("%Y-%m-%d")
                        stock_data = portfolio[stock]['stock_data']['6M_Marketbeat'].get(date, None)
                        price_usd_target = stock_data[list(stock_data.keys())[0]]['ClosingPrice'] if stock_data else None
                        j+=1
                        price_flag = date if price_usd_target else None
                        if price_flag:
                            portfolio[stock]['stock_data']['date_for_calculation'] = price_flag
                total_usd_target = amount * price_usd_target

            price_usd_start = portfolio[stock]['stock_data']['6M_Marketbeat'][startdate_format][list(portfolio[stock]['stock_data']['6M_Marketbeat'][startdate_format].keys())[0]]['ClosingPrice']
            total_usd_start = amount * price_usd_start
            stock_chg = total_usd_target / total_usd_start * 100 - 100 if not total_usd_start == 0 else 0
            yields= defaultdict(dict)
            yields['chg last {} days'.format(per)] = stock_chg
            portfolio[stock]['stock_data']['yield'] = yields
            if withRival:
                # Check whether the similar stocks performed better
                better_list = []
                for rstock in portfolio[stock]['similar_stocks']:
                    rival_chg = check_difference(rstock, rival_portfolio, startdate_format, end_date)
                    if  rival_chg > stock_chg :
                        s_amount = calc_equiv_stock_amount(total_usd_start,rstock,rival_portfolio,startdate_format)
                        new_tripl = [rstock, s_amount ,rival_chg]
                        better_list.append(new_tripl)
                    else:
                        continue
                is_better_dict[(stock, stock_chg)] = better_list

            # Sums up the amounts for the current portfolio performance
            sum_target += total_usd_target
            sum_start += total_usd_start
            chg['{} days'.format(per)] = 100*(sum_target / sum_start)- 100 if sum_start else None
    if withRival:
        return chg, is_better_dict, rival_portfolio
    else:
        return chg

def create_alternative_portfolio(better_dict, portfolio, alt_stock_data ):
    new_portfolio = dict(portfolio)
    new_list_stocks = []
    i = 0
    for stock in better_dict:
        if better_dict[stock]:
            new_portfolio.pop(stock[0])
            def func(p):
                return p[2]
            best = max(better_dict[stock], key = func,default=[0,0,0])
            new_list_stocks.append(best[:2])
            best_stock_name = new_list_stocks[i][0]
            alt_stock_data[best_stock_name]['amount'] = new_list_stocks[i][1]
            new_portfolio[best_stock_name] = alt_stock_data[best_stock_name]
            i += 1
    return new_portfolio

def stocks_statistics(dx_masked):
    # Create series of filtered array
    count_zeros,count_down,count_up= [],[],[]
    end_of_j, j = False, 0
    init_up_count,init_down_count,init_z_count = False,False,False
    z_counter,up_count,down_count = 0,0,0
    while j< len(dx_masked):
        z_counter = 0 if init_z_count else z_counter
        down_count = 0 if init_down_count else down_count
        up_count = 0 if init_up_count else up_count
        if j == len(dx_masked): break
        if not end_of_j:
            if dx_masked[j] == 0.0:
                while (dx_masked[j] == 0.0 and j<=len(dx_masked)-1):
                    z_counter += 1
                    j+=1
                    if j == len(dx_masked): end_of_j = True; break
                if z_counter > 8 : init_z_count = False; init_down_count,init_up_count=True,True
                else: count_zeros.append(z_counter); init_z_count = True
        if not end_of_j:
            if dx_masked[j] < 0:
                while (dx_masked[j] < 0 and j <= len(dx_masked)-1):
                    down_count += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if down_count > 8: init_down_count = False; init_up_count,init_z_count=True,True
                else: count_down.append(down_count); init_down_count = True
        if not end_of_j:
            if dx_masked[j] > 0:
                while (dx_masked[j] > 0 and j <= len(dx_masked)-1):
                    up_count += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if up_count > 8: init_up_count = False; init_z_count,init_down_count=True,True
                else: count_up.append(up_count); init_up_count=True
    if count_up:
        up_med = statistics.median(count_up) if count_up else 0
        up_mean = statistics.mean(count_up)
        up_medg = statistics.median_grouped(count_up)
        up_var = statistics.variance(count_up) if len(count_up)>2 else 0
    else: up_med,up_mean,up_var,up_medg = 0,0,0,0
    if count_zeros:
        zero_med = statistics.median(count_zeros) if count_zeros else 0
        zero_mean = statistics.mean(count_zeros)
        zero_var = statistics.variance(count_zeros) if len(count_zeros)>2 else 0
        zero_medg = statistics.median_grouped(count_zeros)
    else: zero_var,zero_mean,zero_medg,zero_med = 0,0,0,0
    if count_down:
        down_med = statistics.median(count_down) if count_down else 0
        down_mean = statistics.mean(count_down)
        down_var = statistics.variance(count_down) if len(count_down)>2 else 0
        down_medg = statistics.median_grouped(count_down)
    else: down_var,down_mean,down_medg,down_med = 0,0,0,0
    med_arr = numpy.column_stack([[up_med,up_medg,up_mean,up_var,len(count_up)],[zero_med,zero_medg,zero_mean,zero_var,len(count_zeros)],[down_med,down_medg,down_mean,down_var,len(count_down)]])
    stats_df = pd.DataFrame(med_arr,columns= ['Up','Zero','Down'],index=['Median','Grouped Median','Mean Value','Variance','Num of Series'])
    #count_arr = numpy.array(count_down)
    #count_df = pd.DataFrame(count_arr)
    #count_df.plot(title = '{}'.format(portfolio[stock]['stock_data']['name']) ,kind = 'kde')
    #plt.plot(count_up,range(len(count_up)),'.')
   # npcountup=numpy.array(count_up)
   # npx=numpy.linspace(0,4,16)
    #z = numpy.polyfit(npx,npcountup, 10)
    #plt.plot(z)
    #def func(x,a,b,c):
    #    return a * numpy.exp(-b * (x*x)) + c
    #optimizedParameters, pcov = opt.curve_fit(func,npx, npcountup)
    #plt.plot(npx, func(npx, *optimizedParameters), label="fit")
    return stats_df

def calc_stock_dataframe(portfolio,stock,ref_timeframe, period='1M',percentile=97,to_plot=False, is_vol = False):
    deriv_dates = sorted(list(portfolio[stock]['stock_data']['6M_Marketbeat'].keys()))
    deriv_1M_dates = sorted(list(portfolio[stock]['stock_data']['{}'.format(period)].keys()))
    price_sorted: List[Any] = []
    price_1M_sorted: List[Any] = []
    for k in deriv_dates:
        check = portfolio[stock]['stock_data']['6M_Marketbeat'][k].get(stock,None)
        if check:
            price = portfolio[stock]['stock_data']['6M_Marketbeat'][k][stock]['ClosingPrice']
            price_sorted.append(price)
        else:
            price_sorted.append(price_sorted[-1])
    if portfolio[stock]['stock_data']['real_time']:
        deriv_dates.append(today.strftime("%Y-%m-%d"))
        price_sorted.append(list(portfolio[stock]['stock_data']['real_time'].values())[0])
    for d in deriv_1M_dates:
        check2 = portfolio[stock]['stock_data']['{}'.format(period)][d].get('close',None) if not is_vol else portfolio[stock]['stock_data']['{}'.format(period)][d].get('volume',None)
        if check2:
            price1M = portfolio[stock]['stock_data']['{}'.format(period)][d]['close'] if not is_vol else portfolio[stock]['stock_data']['{}'.format(period)][d]['volume']
            price_1M_sorted.append(price1M)
        else:
            price_1M_sorted.append(price_sorted[-1])
    if not is_vol:
        if portfolio[stock]['stock_data']['real_time']:
            deriv_1M_dates.append(today.strftime("%Y-%m-%d"))
            price_1M_sorted.append(list(portfolio[stock]['stock_data']['real_time'].values())[0])
    dx_arr = numpy.diff(price_sorted)
    dx_1M_arr = numpy.diff(price_1M_sorted)
    price_array = numpy.array(price_sorted[1:])
    price_1M_array = numpy.array(price_1M_sorted[1:])
    med = numpy.median(dx_arr)
    per = numpy.percentile(dx_arr,percentile)
    per1M = numpy.percentile(dx_1M_arr,percentile)
    # Create the Scaler object
    #scaler = preprocessing.StandardScaler()
    scaler_MinMax = preprocessing.MinMaxScaler()
    #Create Masking according to the requested Percentiles
    mask_plus = (dx_arr>per)
    mask_minus = (dx_arr<-per)
    mask = numpy.logical_or(mask_minus,mask_plus)
    ones = numpy.ones_like(dx_arr)
    mask = numpy.logical_xor(mask,ones)
    masked_array = numpy.ma.array(dx_arr,mask= mask)
    dx_masked = numpy.ma.filled(masked_array,[0])
    #Masking for 1M
    mask_1M_plus = (dx_1M_arr>per1M)
    mask_1M_minus = (dx_1M_arr<-per1M)
    mask1M = numpy.logical_or(mask_1M_minus,mask_1M_plus)
    ones1M = numpy.ones_like(dx_1M_arr)
    mask1M = numpy.logical_xor(mask1M,ones1M)
    masked_1M_array = numpy.ma.array(dx_1M_arr,mask= mask1M)
    dx_1M_masked = numpy.ma.filled(masked_1M_array,[0])
    # Scaling Prices and Derivatives
    scaled_price = scaler_MinMax.fit_transform(price_array.reshape(-1, 1))
    scaled_dx = scaler_MinMax.fit_transform(dx_masked.reshape(-1, 1))
    scaled_1M_price = scaler_MinMax.fit_transform(price_1M_array.reshape(-1, 1))
    scaled_1M_dx = scaler_MinMax.fit_transform(dx_1M_masked.reshape(-1, 1))
    deriv_dates = deriv_dates[1:]
    deriv_1M_dates = deriv_1M_dates[1:]
    cols = numpy.column_stack((deriv_dates[-ref_timeframe:], dx_masked[-ref_timeframe:], scaled_dx[-ref_timeframe:], price_array[-ref_timeframe:], scaled_price[-ref_timeframe:]))
    df = pd.DataFrame(cols,columns= ['DateTime','Derivatives','Deriv_Norm','Price','Price_Norm'])
    if is_vol:
        cols1M = numpy.column_stack((deriv_1M_dates[-ref_timeframe:], dx_1M_masked[-ref_timeframe:], scaled_1M_dx[-ref_timeframe:], price_1M_array[-ref_timeframe:], scaled_1M_price[-ref_timeframe:]))
    else:
        cols1M = numpy.column_stack((deriv_1M_dates, dx_1M_masked,scaled_1M_dx, price_1M_array,scaled_1M_price))
    df1M = pd.DataFrame(cols1M,columns= ['DateTime','Derivatives','Deriv_Norm','Price','Price_Norm']) if not is_vol else pd.DataFrame(cols1M,columns= ['DateTime','Derivatives','Deriv_Norm','Volume','Volume_Norm'])
    #Casting objects
    df['DateTime'] = df['DateTime'].astype('datetime64[h]')
    df['Deriv_Norm']=df['Deriv_Norm'].astype('float64')
    df['Derivatives']=df['Derivatives'].astype('float64')
    df['Price_Norm']=df['Price_Norm'].astype('float64')
    df['Price']=df['Price'].astype('float64')
    #1M
    df1M['DateTime'] = df1M['DateTime'].astype('datetime64[h]')
    df1M['Deriv_Norm']=df1M['Deriv_Norm'].astype('float64')
    df1M['Derivatives']=df1M['Derivatives'].astype('float64')
    if not is_vol:
        df1M['Price_Norm']=df1M['Price_Norm'].astype('float64')
    else:
        df1M['Volume_Norm']=df1M['Volume_Norm'].astype('float64')
    if not is_vol:
        df1M['Price']=df1M['Price'].astype('float64')
    else:
        df1M['Volume']=df1M['Volume'].astype('float64')
    #df.info()
    if is_vol :
        return df1M
    else:
        return df

def correlate_dict(portfolio,stock,date_only_dict,up_or_down_list):
    hour_dict = {}
    for d in up_or_down_list:
        for hour in date_only_dict[d]:
            hour_parse = parser.parse(hour)
            hour_only = hour_parse.strftime("%H:%M:%S")
            if hour_only not in hour_dict:
                hours_price = []
                hours_price.append((portfolio[stock]['stock_data']['1M'][hour].get('close', None) /
                                    portfolio[stock]['stock_data']['6M'][d + ' 16:00:00'].get('open', None)) - 1)
                hour_dict[hour_only] = hours_price
            else:
                hour_dict[hour_only].append((portfolio[stock]['stock_data']['1M'][hour].get('close', None) /
                                             portfolio[stock]['stock_data']['6M'][d + ' 16:00:00'].get('open',
                                                                                                       None)) - 1)
    close_price_list = [(portfolio[stock]['stock_data']['6M'][date + ' 16:00:00'].get('close', None) /
                         portfolio[stock]['stock_data']['6M'][date + ' 16:00:00'].get('open', None)) - 1 for date in
                        up_or_down_list]
    corr_dict = {}
    for h in hour_dict:
        array1 = numpy.array(hour_dict[h])
        array2 = numpy.array(close_price_list)
        correlate = pearsonr(array1, array2)
        corr_dict[h] = correlate
    return corr_dict
def plot_momentum_series(portfolio,timeframe, is_plot):
    for stock in portfolio:
        price_frame = calc_stock_dataframe(portfolio,stock,timeframe)
        price_month_frame = calc_stock_dataframe(portfolio,stock,timeframe, period="1M")
        volume_frame = calc_stock_dataframe(portfolio,stock,timeframe,period="6M", is_vol=True)
        price_stats = stocks_statistics(price_frame['Derivatives'])
        volume_stats = stocks_statistics(volume_frame['Derivatives'])
        #####Continue HERE~!
        if is_plot:
            #if not ((price_stats.iloc[:3]['Down']==1).all() or (price_stats.iloc[:3]['Up']==1).all() or (price_stats.iloc[:3]['Down']==1).all()):
                #x = numpy.linspace(0, 4, 10)
                #price_stats.iloc[:3].plot.kde(title = '{}'.format(portfolio[stock]['stock_data']['name']) ,xticks=x)
            #if (price_stats.iloc[3]>2).any() :
                #stock_frame = calc_stock_dataframe(portfolio,stock,int(timeframe/2),70)
                #price_stats = stocks_statistics(stock_frame['Derivatives'])
            # Plot Option for manual analysis
            temp_df = pd.DataFrame.copy(price_frame)
            temp_df_vol = pd.DataFrame.copy(volume_frame)
            corr = temp_df.corrwith(temp_df_vol)
            temp_df.set_index(['DateTime'], inplace=True)
            ax = temp_df[['Price_Norm']].plot(title='{}\n Price vs. Volume\n'.format(portfolio[stock]['stock_data']['name'],temp_df[['Price_Norm']]))
            temp_df_vol.set_index(['DateTime'], inplace=True)
            temp_df_vol[['Volume_Norm']].plot(ax=ax)
            temp_df['Volume_Norm'] = temp_df_vol['Volume_Norm']
            print('Stock {}\n '.format(stock),corr)
        days_list = sorted(list(portfolio[stock]['stock_data']['6M'].keys()))
        hour_list = sorted(list(portfolio[stock]['stock_data']['1M'].keys()))
        date_only_dict = {}
        #Creates Dict - each day with a 30 min delta (except 16:00, closing price)
        for hour in hour_list:
            hour_date = parser.parse(hour)
            date_only = hour_date.strftime("%Y-%m-%d")
            if date_only not in date_only_dict:
                hours = []
                hours.append(hour)
                date_only_dict[date_only] = hours
            else:
                hours.append(hour)
                date_only_dict[date_only] = hours
        up_list,down_list,hour_list = [],[],[]
        #Creates Down and Up Lists
        for day in date_only_dict:
            if portfolio[stock]['stock_data']['6M'][day+' 16:00:00'].get('close',None) > portfolio[stock]['stock_data']['6M'][day+' 16:00:00'].get('open',None):
                up_list.append(day)
            elif portfolio[stock]['stock_data']['6M'][day+' 16:00:00'].get('close',None) < portfolio[stock]['stock_data']['6M'][day+' 16:00:00'].get('open',None):
                down_list.append(day)
        corr_dict_up = correlate_dict(portfolio,stock,date_only_dict,up_list)
        corr_dict_down = correlate_dict(portfolio, stock, date_only_dict, down_list)
        portfolio[stock]['stock_data']['hourly_correlation'] = {'up':corr_dict_up}
        portfolio[stock]['stock_data']['hourly_correlation']['down'] = corr_dict_down
            #price_chg = check_difference(stock, portfolio, )
            #if price_chg > 0 :
                #for price in price_frame[]:




##Main
#Checks Test Purpose or Not?
input_str = input("Test Purpose: Y/N ")
if not re.match("^[ynYN]*$", input_str):
    print ("Error! Only letters y/n allowed!")
    sys.exit()
elif len(input_str) > 1:
    print ("Error! Only 1 characters allowed!")
    sys.exit()
today = datetime.datetime.today()
stocks = [['XLE',55],['XLC',39],['XLF',153],['RYAAY',54],['UAL',163],['SOXL',21]]
alternative, portfolio = defaultdict(dict), defaultdict(dict)
is_test = True if input_str == ('y' or 'Y') else False
if is_test :
##Loads portfolio from file - Test purpose
    f = open("22.4.json","r")
    portfolio = json.load(f)
else:
    ##Save Portfolio to file
    portfolio = create_Portfolio(stocks)
    json_file = json.dumps(portfolio)
    f = open("22.4.json","w")
    f.write(json_file)
    f.close()

chg, better_dict, alt_stock_data = calc_performance(portfolio,is_test,7)
new_portfolio = create_alternative_portfolio(better_dict, portfolio,alt_stock_data)
new_chg = calc_performance(new_portfolio,is_test,7,withRival=False)
#Timeframe for Relative Calculation
timeframe = 60
is_plot = False
plot_momentum_series(portfolio,timeframe,is_plot)
f.close()

print (1)

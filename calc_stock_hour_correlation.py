import datetime
import numpy
from dateutil import parser
from scipy.stats import pearsonr


def correlate_dict(portfolio, stock, date_only_dict, up_or_down_list):
    hour_dict = {}
    for d in up_or_down_list:
        for hour in date_only_dict[d]:
            hour_parse = parser.parse(hour)
            hour_only = hour_parse.strftime("%H:%M:%S")
            # Relative change - price at specific hour/open price
            if hour_only not in hour_dict:
                hours_price = []
                hours_price.append((portfolio[stock]['stock_data']['1M'][hour].get('close', None) /
                                    portfolio[stock]['stock_data']['6M'][d + ' 16:00:00'].get('open', None)) - 1)
                hour_dict[hour_only] = hours_price
            else:
                hour_dict[hour_only].append((portfolio[stock]['stock_data']['1M'][hour].get('close', None) /
                                             portfolio[stock]['stock_data']['6M'][d + ' 16:00:00'].get('open',
                                                                                                       None)) - 1)
    # Close / Open price
    close_price_list = [(portfolio[stock]['stock_data']['6M'][date + ' 16:00:00'].get('close', None) /
                         portfolio[stock]['stock_data']['6M'][date + ' 16:00:00'].get('open', None)) - 1 for date in
                        up_or_down_list]
    corr_dict = {}
    for h in hour_dict:
        array1 = numpy.array(hour_dict[h])
        array2 = numpy.array(close_price_list)
        if len(array1) == len(array2) and len(array1) >= 2:
            correlate = pearsonr(array1, array2)
        else:
            correlate = 'N/A'
        corr_dict[h] = correlate
    return corr_dict


def stock_hour_correlation(portfolio, stock, window=30):
    days_list = sorted(list(portfolio[stock]['stock_data']['6M'].keys()))
    hour_list = sorted(list(portfolio[stock]['stock_data']['1M'].keys()))
    base = datetime.datetime.today()
    requested_date_list = [(base - datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range(window)]
    date_only_dict = {}
    # Creates Dict - each day with a 30 min delta (except 16:00, closing price)
    for hour in hour_list:
        hour_date = parser.parse(hour)
        date_only = hour_date.strftime("%Y-%m-%d")
        if date_only not in requested_date_list: continue
        if date_only not in date_only_dict:
            hours = []
            hours.append(hour)
            date_only_dict[date_only] = hours
        else:
            hours.append(hour)
            date_only_dict[date_only] = hours
    up_list, down_list, hour_list = [], [], []
    # Creates Down and Up Lists
    try:
        for day in date_only_dict:
            if portfolio[stock]['stock_data']['6M'].get(day + ' 16:00:00', None):
                if portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None):
                    if portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) > \
                            portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                        up_list.append(day)
                    elif portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) < \
                            portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                        down_list.append(day)
    except TypeError:
        print('Error in {} stock, date - {}'.format(stock, day))
    corr_dict_up = correlate_dict(portfolio, stock, date_only_dict, up_list)
    corr_dict_down = correlate_dict(portfolio, stock, date_only_dict, down_list)
    portfolio[stock]['stock_data']['hourly_correlation'] = {'up': corr_dict_up}
    portfolio[stock]['stock_data']['hourly_correlation']['down'] = corr_dict_down

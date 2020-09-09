from typing import List, Any
import datetime
import numpy as np
from sklearn import preprocessing
import pandas as pd
import statistics as stat
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
        array1 = np.array(hour_dict[h])
        array2 = np.array(close_price_list)
        if len(array1) == len(array2) and len(array1) >= 2:
            correlate = pearsonr(array1, array2)
        else:
            correlate = 'N/A'
        corr_dict[h] = correlate
    return corr_dict


def stocks_stat(dx_masked, dx_normal):
    # Create series of filtered array
    count_zeros, count_down, count_up = [], [], []
    end_of_j, j = False, 0
    init_up_count, init_down_count, init_z_count = False, False, False
    z_counter, up_count, down_count = 0, 0, 0
    while j < len(dx_masked):
        z_counter = 0 if init_z_count else z_counter
        down_count = 0 if init_down_count else down_count
        up_count = 0 if init_up_count else up_count
        if j == len(dx_masked): break
        if not end_of_j:
            if dx_masked[j] == 0.0:
                z_counter += 1
                while (dx_normal[j] == 0.0 and j <= len(dx_normal) - 1):
                    z_counter += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if z_counter > 8:
                    init_z_count = False;
                    init_down_count, init_up_count = True, True
                else:
                    count_zeros.append(z_counter - 1);
                    init_z_count = True
                if z_counter == 1: j += 1; continue
        if not end_of_j:
            if dx_masked[j] < 0:
                down_count += 1
                while (dx_normal[j] < 0 and j <= len(dx_normal) - 1):
                    down_count += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if down_count > 8:
                    init_down_count = False;
                    init_up_count, init_z_count = True, True
                else:
                    count_down.append(down_count - 1);
                    init_down_count = True
                if down_count == 1: j += 1; continue
        if not end_of_j:
            if dx_masked[j] > 0:
                up_count += 1
                while (dx_normal[j] > 0 and j <= len(dx_normal) - 1):
                    up_count += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if up_count > 8:
                    init_up_count = False;
                    init_z_count, init_down_count = True, True
                else:
                    count_up.append(up_count - 1);
                    init_up_count = True
                if up_count == 1: j += 1; continue
    if count_up:
        up_med = stat.median(count_up) if count_up else 0
        up_mean = stat.mean(count_up)
        up_medg = stat.median_grouped(count_up)
        up_var = stat.variance(count_up) if len(count_up) > 2 else 0
    else:
        up_med, up_mean, up_var, up_medg = 0, 0, 0, 0
    if count_zeros:
        zero_med = stat.median(count_zeros) if count_zeros else 0
        zero_mean = stat.mean(count_zeros)
        zero_var = stat.variance(count_zeros) if len(count_zeros) > 2 else 0
        zero_medg = stat.median_grouped(count_zeros)
    else:
        zero_var, zero_mean, zero_medg, zero_med = 0, 0, 0, 0
    if count_down:
        down_med = stat.median(count_down) if count_down else 0
        down_mean = stat.mean(count_down)
        down_var = stat.variance(count_down) if len(count_down) > 2 else 0
        down_medg = stat.median_grouped(count_down)
    else:
        down_var, down_mean, down_medg, down_med = 0, 0, 0, 0
    med_arr = np.column_stack([[up_med, up_medg, up_mean, up_var, len(count_up)],
                               [zero_med, zero_medg, zero_mean, zero_var, len(count_zeros)],
                               [down_med, down_medg, down_mean, down_var, len(count_down)]])
    stats_df = pd.DataFrame(med_arr, columns=['Up', 'Zero', 'Down'],
                            index=['Median', 'Grouped Median', 'Mean Value', 'Variance', 'Num of Series'])
    # count_arr = np.array(count_down)
    # count_df = pd.DataFrame(count_arr)
    # count_df.plot(title = '{}'.format(portfolio[stock]['stock_data']['name']) ,kind = 'kde')
    # plt.plot(count_up,range(len(count_up)),'.')
    # npcountup=np.array(count_up)
    # npx=np.linspace(0,4,16)
    # z = np.polyfit(npx,npcountup, 10)
    # plt.plot(z)
    # def func(x,a,b,c):
    #    return a * np.exp(-b * (x*x)) + c
    # optimizedParameters, pcov = opt.curve_fit(func,npx, npcountup)
    # plt.plot(npx, func(npx, *optimizedParameters), label="fit")
    return stats_df


def create_dataframe(portfolio, stock, ref_timeframe, period='1M', percentile=90, to_plot=False, is_vol=False):
    today = datetime.datetime.today()
    deriv_dates = sorted(list(portfolio[stock]['stock_data']['6M_Marketbeat'].keys()))
    deriv_1M_dates = sorted(list(portfolio[stock]['stock_data']['{}'.format(period)].keys()))
    price_sorted: List[Any] = []
    price_1M_sorted: List[Any] = []
    for k in deriv_dates:
        check = portfolio[stock]['stock_data']['6M_Marketbeat'][k].get(stock, None)
        if check:
            price = portfolio[stock]['stock_data']['6M_Marketbeat'][k][stock]['ClosingPrice']
            price_sorted.append(price)
        else:
            price_sorted.append(price_sorted[-1])
    if portfolio[stock]['stock_data']['real_time']:
        deriv_dates.append(today.strftime("%Y-%m-%d"))
        price_sorted.append(list(portfolio[stock]['stock_data']['real_time'].values())[0])
    for d in deriv_1M_dates:
        check2 = portfolio[stock]['stock_data']['{}'.format(period)][d].get('close', None) if not is_vol else \
            portfolio[stock]['stock_data']['{}'.format(period)][d].get('volume', None)
        if check2:
            price1M = portfolio[stock]['stock_data']['{}'.format(period)][d]['close'] if not is_vol else \
                portfolio[stock]['stock_data']['{}'.format(period)][d]['volume']
            price_1M_sorted.append(price1M)
        else:
            price_1M_sorted.append(price_sorted[-1])
    if not is_vol:
        if portfolio[stock]['stock_data']['real_time']:
            deriv_1M_dates.append(today.strftime("%Y-%m-%d"))
            price_1M_sorted.append(list(portfolio[stock]['stock_data']['real_time'].values())[0])
    dx_arr = np.diff(price_sorted)
    dx_1M_arr = np.diff(price_1M_sorted)
    price_array = np.array(price_sorted[1:])
    price_1M_array = np.array(price_1M_sorted[1:])
    med = np.median(dx_arr)
    per = np.percentile(dx_arr, percentile)
    per1M = np.percentile(dx_1M_arr, percentile)
    # Create the Scaler object
    # scaler = preprocessing.StandardScaler()
    scaler_MinMax = preprocessing.MinMaxScaler()
    # Create Masking according to the requested Percentiles
    mask_plus = (dx_arr > per)
    mask_minus = (dx_arr < -per)
    mask = np.logical_or(mask_minus, mask_plus)
    ones = np.ones_like(dx_arr)
    mask = np.logical_xor(mask, ones)
    masked_array = np.ma.array(dx_arr, mask=mask)
    dx_masked = np.ma.filled(masked_array, [0])
    # Masking for 1M
    mask_1M_plus = (dx_1M_arr > per1M)
    mask_1M_minus = (dx_1M_arr < -per1M)
    mask1M = np.logical_or(mask_1M_minus, mask_1M_plus)
    ones1M = np.ones_like(dx_1M_arr)
    mask1M = np.logical_xor(mask1M, ones1M)
    masked_1M_array = np.ma.array(dx_1M_arr, mask=mask1M)
    dx_1M_masked = np.ma.filled(masked_1M_array, [0])
    # Scaling Prices and Derivatives
    scaled_price = scaler_MinMax.fit_transform(price_array.reshape(-1, 1))
    scaled_dx = scaler_MinMax.fit_transform(dx_masked.reshape(-1, 1))
    scaled_1M_price = scaler_MinMax.fit_transform(price_1M_array.reshape(-1, 1))
    scaled_1M_dx = scaler_MinMax.fit_transform(dx_1M_masked.reshape(-1, 1))
    deriv_dates = deriv_dates[1:]
    deriv_1M_dates = deriv_1M_dates[1:]
    cols = np.column_stack((deriv_dates[-ref_timeframe:], dx_masked[-ref_timeframe:], scaled_dx[-ref_timeframe:],
                            price_array[-ref_timeframe:], scaled_price[-ref_timeframe:], dx_arr[-ref_timeframe:]))
    df = pd.DataFrame(cols, columns=['DateTime', 'Derivatives', 'Deriv_Norm', 'Price', 'Price_Norm', 'Unmasked'])
    if is_vol:
        cols1M = np.column_stack((deriv_1M_dates[-ref_timeframe:], dx_1M_masked[-ref_timeframe:],
                                  scaled_1M_dx[-ref_timeframe:], price_1M_array[-ref_timeframe:],
                                  scaled_1M_price[-ref_timeframe:], dx_arr[-ref_timeframe:]))
    else:
        cols1M = np.column_stack((deriv_1M_dates[-ref_timeframe:], dx_1M_masked[-ref_timeframe:],
                                  scaled_1M_dx[-ref_timeframe:], price_1M_array[-ref_timeframe:],
                                  scaled_1M_price[-ref_timeframe:], dx_arr[-ref_timeframe:]))
    df1M = pd.DataFrame(cols1M, columns=['DateTime', 'Derivatives', 'Deriv_Norm', 'Price', 'Price_Norm',
                                         'Unmasked']) if not is_vol else pd.DataFrame(cols1M, columns=['DateTime',
                                                                                                       'Derivatives',
                                                                                                       'Deriv_Norm',
                                                                                                       'Volume',
                                                                                                       'Volume_Norm',
                                                                                                       'Unmasked'])
    # Casting objects
    df['DateTime'] = df['DateTime'].astype('datetime64[h]')
    df['Deriv_Norm'] = df['Deriv_Norm'].astype('float64')
    df['Derivatives'] = df['Derivatives'].astype('float64')
    df['Price_Norm'] = df['Price_Norm'].astype('float64')
    df['Price'] = df['Price'].astype('float64')
    df['Unmasked'] = df['Unmasked'].astype('float64')
    # 1M
    df1M['DateTime'] = df1M['DateTime'].astype('datetime64[h]')
    df1M['Deriv_Norm'] = df1M['Deriv_Norm'].astype('float64')
    df1M['Derivatives'] = df1M['Derivatives'].astype('float64')
    df1M['Unmasked'] = df1M['Unmasked'].astype('float64')
    if not is_vol:
        df1M['Price_Norm'] = df1M['Price_Norm'].astype('float64')
    else:
        df1M['Volume_Norm'] = df1M['Volume_Norm'].astype('float64')
    if not is_vol:
        df1M['Price'] = df1M['Price'].astype('float64')
    else:
        df1M['Volume'] = df1M['Volume'].astype('float64')
    # df.info()
    if is_vol:
        return df1M
    else:
        return df


def create_stock_dataframe(portfolio, timeframe, is_plot=False):
    for stock in portfolio:
        price_frame = create_dataframe(portfolio, stock, timeframe)
        price_month_frame = create_dataframe(portfolio, stock, timeframe, period="1M")
        volume_frame = create_dataframe(portfolio, stock, timeframe, period="6M", is_vol=True)
        price_stats = stocks_stat(price_frame['Derivatives'], price_frame['Unmasked'])
        volume_stats = stocks_stat(volume_frame['Derivatives'], volume_frame['Unmasked'])
        portfolio[stock]['stock_stats'] = {'volume_stats': volume_stats, 'price_stats': price_stats}
        #####Continue HERE~!
        price_fft = np.fft.rfft(price_month_frame.Price)
        n_samples = price_fft.size
        time_delta = 3600
        sample_rate = n_samples / time_delta
        freq = np.fft.rfftfreq(n_samples, d=sample_rate)

        if is_plot:
            # if not ((price_stats.iloc[:3]['Down']==1).all() or (price_stats.iloc[:3]['Up']==1).all() or (price_stats.iloc[:3]['Down']==1).all()):
            # x = np.linspace(0, 4, 10)
            # price_stats.iloc[:3].plot.kde(title = '{}'.format(portfolio[stock]['stock_data']['name']) ,xticks=x)
            # if (price_stats.iloc[3]>2).any() :
            # stock_frame = create_dataframe(portfolio,stock,int(timeframe/2),70)
            # price_stats = stocks_stat(stock_frame['Derivatives'])
            # Plot Option for manual analysis
            temp_df = pd.DataFrame.copy(price_frame)
            temp_df_vol = pd.DataFrame.copy(volume_frame)
            corr = temp_df.corrwith(temp_df_vol)
            temp_df.set_index(['DateTime'], inplace=True)
            ax = temp_df[['Price_Norm']].plot(
                title='{}\n Price vs. Volume\n'.format(portfolio[stock]['stock_data']['name'], temp_df[['Price_Norm']]))
            temp_df_vol.set_index(['DateTime'], inplace=True)
            temp_df_vol[['Volume_Norm']].plot(ax=ax)
            temp_df['Volume_Norm'] = temp_df_vol['Volume_Norm']
            print('Stock {}\n '.format(stock), corr)
        days_list = sorted(list(portfolio[stock]['stock_data']['6M'].keys()))
        hour_list = sorted(list(portfolio[stock]['stock_data']['1M'].keys()))
        date_only_dict = {}
        # Creates Dict - each day with a 30 min delta (except 16:00, closing price)
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
        up_list, down_list, hour_list = [], [], []
        # Creates Down and Up Lists
        for day in date_only_dict:
            if portfolio[stock]['stock_data']['6M'].get(day + ' 16:00:00', None):
                if portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) and \
                        portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                    if portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) > \
                            portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                        up_list.append(day)
                    elif portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) < \
                            portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                        down_list.append(day)
        corr_dict_up = correlate_dict(portfolio, stock, date_only_dict, up_list)
        corr_dict_down = correlate_dict(portfolio, stock, date_only_dict, down_list)
        portfolio[stock]['stock_data']['hourly_correlation'] = {'up': corr_dict_up}
        portfolio[stock]['stock_data']['hourly_correlation']['down'] = corr_dict_down
        # price_chg = check_difference(stock, portfolio, )
        # if price_chg > 0 :
        # for price in price_frame[]:

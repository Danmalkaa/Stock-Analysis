def change_from_high(alt_stock_data, key, is_ptarget=False):
    if not is_ptarget:
        high_price = alt_stock_data[key]['Tiprank']['ticker_data']['historicalHighLow'][-1]['high']
    else:
        high_price = alt_stock_data[key]['Tiprank']['ticker_data']['analystPriceTarget']
    if alt_stock_data[key]['stock_data'].get('real_time', None):
        current_price = list(alt_stock_data[key]['stock_data']['real_time'].values())[0]
    else:
        current_price = \
            alt_stock_data[key]['stock_data']['1M'][sorted(alt_stock_data[key]['stock_data']['1M'].keys())[-1]][
                'close']
    if not current_price or not high_price:
        return 'N/A'
    else:
        if not is_ptarget:
            return (current_price / high_price - 1) * 100
        else:
            return (high_price / current_price - 1) * 100


def create_accuracy_dict(alt_stock_data):
    good_accuracy = {'70%-80%': [], '80%-90%': [], '>90%': []}
    for key in alt_stock_data.keys():
        if alt_stock_data[key]['Tiprank'].get('Tip_accuracy', None):
            if alt_stock_data[key]['Tiprank'].get('Tip_accuracy', None)[0] >= 70 and \
                    alt_stock_data[key]['Tiprank'].get('Tip_accuracy', None)[0] < 80:
                good_accuracy['70%-80%'].append(
                    [key, 'Stock Score: {}'.format(alt_stock_data[key]['Tiprank']['tipranksStockScore']['score']),
                     '% Change from 2020 High: {}'.format(change_from_high(alt_stock_data, key)),
                     '% Change from Price Target: {}'.format(change_from_high(alt_stock_data, key,
                                                                              is_ptarget=True))])  # append({key:alt_stock_data[key]})
            if alt_stock_data[key]['Tiprank'].get('Tip_accuracy', None)[0] >= 80 and \
                    alt_stock_data[key]['Tiprank'].get('Tip_accuracy', None)[0] < 90:
                good_accuracy['80%-90%'].append(
                    [key, 'Stock Score: {}'.format(alt_stock_data[key]['Tiprank']['tipranksStockScore']['score']),
                     '% Change from 2020 High: {}'.format(change_from_high(alt_stock_data, key)),
                     '% Change from Price Target: {}'.format(
                         change_from_high(alt_stock_data, key, is_ptarget=True))])
            if alt_stock_data[key]['Tiprank'].get('Tip_accuracy', None)[0] >= 90:
                good_accuracy['>90%'].append(
                    [key, 'Stock Score: {}'.format(alt_stock_data[key]['Tiprank']['tipranksStockScore']['score']),
                     '% Change from 2020 High: {}'.format(change_from_high(alt_stock_data, key)),
                     '% Change from Price Target: {}'.format(change_from_high(alt_stock_data, key,
                                                                              is_ptarget=True))])  # append({key:{'Stock Score':alt_stock_data[key]['Tiprank']['tipranksStockScore']['score'],'% Change from 2020 High':change_from_high(alt_stock_data,key)}})
    # good_accuracy['70%-80%'].sort(key=lambda x: x[list(good_accuracy['70%-80%'][0].keys())[0]]['Stock Score'],reverse=True)
    good_accuracy['70%-80%'].sort(key=lambda x: x[1], reverse=True)
    good_accuracy['80%-90%'].sort(key=lambda x: x[1], reverse=True)
    good_accuracy['>90%'].sort(key=lambda x: x[1], reverse=True)
    return good_accuracy


def create_alternative_portfolio(portfolio, alt_stock_data):
    good_acc_dict, alt_acc_dict = {}, {}
    good_acc_dict = create_accuracy_dict(portfolio)
    alt_acc_dict = create_accuracy_dict(alt_stock_data)
    for key in alt_acc_dict:
        good_acc_dict[key] = alt_acc_dict[key]
    return good_acc_dict

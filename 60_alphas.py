"""
60 formulaic alphas in python from the 101 Formulaic Alphas paper (2015) by Zura Kakushadze

Quantigic® Solutions LLC: https://www.quantigic.com/

We present explicit formulas – that are also computer code – for
101 real-life quantitative trading alphas. Their average holding
period approximately ranges 0.6-6.4 days. The average pair-wise
correlation of these alphas is low, 15.9%. The returns are strongly
correlated with volatility, but have no significant dependence on
turnover, directly confirming an earlier result based on a more
indirect empirical analysis. We further find empirically that
turnover has poor explanatory power for alpha correlations.
"""
import numpy as np
import pandas as pd

#Pre-defined Proprietary Functions
def correlation(x, y, d):
    return np.corrcoef(x[-d:], y[-d:])[0, 1]

def covariance(x, y, d):
    return np.cov(x[-d:], y[-d:])[0, 1]

def scale(x, a=1):    #rescaled x such that sum(abs(x)) = a (the default is a = 1)
    return x / abs(x).sum() * a

def decay_linear(x, d):
    weights = np.linspace(1, 0, d)
    return (x[-d:] * weights).mean()

def ts_min(x, d):
    return x[-d:].min()

def ts_max(x, d):
    return x[-d:].max()

def ts_rank(x, d):
    return np.argsort(x[-d:])[::-1].argsort()[-1] + 1

def ts_argmax(x, d):    #take a time-series x and a lookback window d as inputs, and return an array of the indices at which the max of x within the window d occurred
    max_vals = ts_max(x, d)
    argmax_vals = [i for i, val in enumerate(x) if val == max_vals[i]]
    return argmax_vals

def ts_argmin(x, d):    ##take a time-series x and a lookback window d as inputs, and return an array of the indices at which the min of x within the window d occurred
    min_vals = ts_min(x, d)
    argmin_vals = [i for i, val in enumerate(x) if val == min_vals[i]]
    return argmin_vals

def indneutralize(x, g):
    groups = np.unique(g)
    x_neutralized = np.zeros(x.shape)
    for group in groups:
        group_mask = g == group
        group_mean = np.mean(x[group_mask])
        x_neutralized[group_mask] = x[group_mask] - group_mean
    return x_neutralized

def delta(x, d):
    return x - np.roll(x, d)

def rank(x):    #x being a pandas df or a column of df
    return x.rank(axis=0, pct=True)

def signedpower(x, a):
    return x**a

def stddev(x, d):
    return np.std(x[-d:])

#Alpha Formula Functions
def alpha_1(returns, close):
    std_dev = stddev(returns, 20)
    signed_power = np.where(returns < 0, std_dev, close) ** 2
    ranks = rank(ts_argmax(signed_power, 5))
    return ranks - 0.5

def alpha_2(volume, close, open):
    delta_log_volume = delta(np.log(volume), 2)
    close_open_ratio = (close - open) / open
    return -1 * correlation(rank(delta_log_volume), rank(close_open_ratio), 6)

def alpha_3(open, volume):
    rank_open = rank(open)
    rank_volume = rank(volume)
    corr = correlation(rank_open, rank_volume, 10)
    return -corr

def alpha_4(returns):
    low = returns['low']
    return -1 * ts_rank(rank(low), 9)

def alpha_5(close, open, vwap):
    factor1 = rank(open - (np.sum(vwap[-9:]) / 10))
    factor2 = -1 * np.abs(rank(close - vwap))
    return factor1 * factor2

def alpha_6(open, volume):
    corr = correlation(open, volume, 10)
    return -corr

def alpha_7(close, returns):
    volume = returns['volume']
    adv20 = returns['adv20']
    delta_close = delta(close, 7)
    if adv20 < volume:
        return -1 * ts_rank(np.abs(delta_close), 60) * np.sign(delta_close)
    else:
        return -1 * 1

def alpha_8(returns, open):
    sum_open = np.sum(open[-4:])
    sum_returns = np.sum(returns[-4:])
    sum_open_sum_returns = sum_open * sum_returns
    delayed_sum_open_sum_returns = np.roll(sum_open_sum_returns, 10)
    ranked = -1 * rank(delayed_sum_open_sum_returns - sum_open_sum_returns)
    return ranked

def alpha_9(close):
    delta_close = delta(close, 1)
    ts_min_delta_close = ts_min(delta_close, 5)
    ts_max_delta_close = ts_max(delta_close, 5)
    
    if ts_min_delta_close > 0:
        return delta_close
    elif ts_max_delta_close < 0:
        return delta_close
    else:
        return -1 * delta_close

def alpha_10(close):
    def __calc__(close):
        delta_close = delta(close, 1)
        ts_min_delta_close = ts_min(delta_close, 4)
        ts_max_delta_close = ts_max(delta_close, 4)

        if ts_min_delta_close > 0:
            return delta_close
        elif ts_max_delta_close < 0:
            return delta_close
        else:
            return -1 * delta_close

    return rank(__calc__(close))

def alpha_11(close, vwap, volume):
    vwap_close = vwap - close
    rank_vwap_close_max = rank(ts_max(vwap_close, 3))
    rank_vwap_close_min = rank(ts_min(vwap_close, 3))
    rank_volume_delta = rank(delta(volume, 3))
    return (rank_vwap_close_max + rank_vwap_close_min) * rank_volume_delta

def alpha_12(close, volume):
    return np.sign(delta(volume, 1)) * (-1 * delta(close, 1))

def alpha_13(close, volume):
    rank_close = rank(close)
    rank_volume = rank(volume)
    cov_rank_close_rank_volume = covariance(rank_close, rank_volume, 5)
    return -1 * rank(cov_rank_close_rank_volume)

def alpha_14(returns, open, volume):
    rank_returns_delta = rank(delta(returns, 3))
    corr_open_volume = correlation(open, volume, 10)
    return (-1 * rank_returns_delta) * corr_open_volume

def alpha_15(high, volume):
    rank_high = rank(high)
    rank_volume = rank(volume)
    corr_rank_high_rank_volume = correlation(rank_high, rank_volume, 3)
    rank_corr_rank_high_rank_volume = rank(corr_rank_high_rank_volume)
    return -1 * np.sum(rank_corr_rank_high_rank_volume[-2:])

def alpha_16(high, volume):
    rank_high = rank(high)
    rank_volume = rank(volume)
    cov_rank_high_rank_volume = covariance(rank_high, rank_volume, 10)
    return -1 * rank(cov_rank_high_rank_volume)

def alpha_17(close, volume, adv20):
    ts_rank_close_10 = np.argsort(np.argsort(close, axis=0), axis=0)
    rank_ts_rank_close = np.argsort(ts_rank_close_10, axis=0)
    delta_close_1 = delta(close, 1)
    delta_delta_close = delta(delta_close_1, 1)
    rank_delta_delta_close = np.argsort(delta_delta_close, axis=0)
    volume_per_adv20 = volume / adv20
    ts_rank_volume_per_adv20 = np.argsort(np.argsort(volume_per_adv20, axis=0), axis=0)
    rank_ts_rank_volume_per_adv20 = np.argsort(ts_rank_volume_per_adv20, axis=0)
    return -1 * rank_ts_rank_close * rank_delta_delta_close * rank_ts_rank_volume_per_adv20

def alpha_18(close, open):
    abs_close_open = np.abs(close - open)
    stddev_abs_close_open = stddev(abs_close_open, 5)
    sum_abs_close_open = stddev_abs_close_open + (close - open)
    corr_close_open = correlation(close, open, 10)
    rank_corr_close_open = np.argsort(corr_close_open, axis=0)
    return -1 * rank_corr_close_open

def alpha_19(close, returns):
    delay_close_7 = np.roll(close, 7, axis=0)
    delta_close_7 = delta(close, 7)
    sign_close_delay_close_7_delta_close_7 = np.sign((close - delay_close_7) + delta_close_7)
    returns_sum_250 = np.sum(returns[-249:])
    rank_returns_sum_250 = rank(1 + returns_sum_250)
    return -1 * sign_close_delay_close_7_delta_close_7 * (1 + rank_returns_sum_250)

def alpha_20(open, high, low, close):
    delay_high_1 = np.roll(high, 1, axis=0)
    delay_close_1 = np.roll(close, 1, axis=0)
    delay_low_1 = np.roll(low, 1, axis=0)
    rank_open_delay_high_1 = np.argsort(open - delay_high_1, axis=0)
    rank_open_delay_close_1 = np.argsort(open - delay_close_1, axis=0)
    rank_open_delay_low_1 = np.argsort(open - delay_low_1, axis=0)
    return -1 * rank_open_delay_high_1 * rank_open_delay_close_1 * rank_open_delay_low_1

def alpha_21(close, volume, adv20):
    sum_close_8 = np.sum(close[-7:]) / 8
    sum_close_2 = np.sum(close[-1:]) / 2
    stddev_close_8 = stddev(close,8)
    return np.where((sum_close_8 + stddev_close_8) < sum_close_2, -1, np.where(sum_close_2 < (sum_close_8 - stddev_close_8), 1, np.where((1 < (volume / adv20)) | ((volume / adv20) == 1), 1, -1)))

def alpha_22(high, volume, close):
    delta_corr = delta(correlation(high, volume, 5), 5)
    return -1 * delta_corr * rank(stddev(close, 20))

def alpha_23(high):
    sum_high = np.sum(high[-19:]) / 20
    delta_high = delta(high, 2)
    return np.where((sum_high < high), -1 * delta_high, 0)

def alpha_24(close):
    sum_close = np.sum(close[-99:]) / 100
    delta_sum_close = delta(sum_close, 100)
    delta_close = delta(close, 3)
    ts_min_close = ts_min(close, 100)
    return np.where(((delta_sum_close / np.roll(close, 100)) < 0.05) | ((delta_sum_close / np.roll(close, 100)) == 0.05),
                     -1 * (close - ts_min_close),
                     -1 * delta_close)

def alpha_25(returns, adv20, vwap, high, close):
    return rank((((-1 * returns) * adv20) * vwap) * (high - close))

def alpha_26(volume, high):
    ts_rank_volume = ts_rank(volume, 5)
    ts_rank_high = ts_rank(high, 5)
    correlation_ts = correlation(ts_rank_volume, ts_rank_high, 5)
    return -1 * ts_max(correlation_ts, 3)

def alpha_27(volume, vwap):
    rank_volume = rank(volume)
    rank_vwap = rank(vwap)
    correlation_rank = correlation(rank_volume, rank_vwap, 6)
    sum_correlation_rank = np.sum(correlation_rank[-1:])
    rank_sum_correlation_rank = rank(sum_correlation_rank / 2)
    return np.where(0.5 < rank_sum_correlation_rank, -1, 1)

def alpha_28(adv20, low, high, close):
    correlation_adv = correlation(adv20, low, 5)
    mean_high_low = (high + low) / 2
    return scale((correlation_adv + mean_high_low) - close)

def alpha_30(close, volume):
    sign_close_delay_1 = np.sign(close - np.roll(close, 1))
    sign_delay_1_delay_2 = np.sign(np.roll(close, 1) - np.roll(close, 2))
    sign_delay_2_delay_3 = np.sign(np.roll(close, 2) - np.roll(close, 3))
    rank_sign = 1.0 - rank(sign_close_delay_1 + sign_delay_1_delay_2 + sign_delay_2_delay_3)
    return rank_sign * np.sum(volume[-4:]) / np.sum(volume[-19:])

def alpha_31(close, low, adv20):
    rank_delta_close = rank(delta(close, 10))
    rank_rank_delta_close = rank(rank_delta_close)
    decay_linear_rank = decay_linear(-1 * rank_rank_delta_close, 10)
    rank_decay_linear = rank(decay_linear_rank)
    rank_negative_delta_close = rank(-1 * delta(close, 3))
    correlation_adv20_low = correlation(adv20, low, 12)
    sign_scale_correlation = np.sign(scale(correlation_adv20_low))
    return rank_decay_linear + rank_negative_delta_close + sign_scale_correlation

def alpha_32(close, vwap):
    scale_close_sum = scale((np.sum(close[-6:]) / 7) - close)
    scale_correlation_vwap_close = 20 * scale(correlation(vwap, np.roll(close, 5), 230))
    return scale_close_sum + scale_correlation_vwap_close

def alpha_33(open_, close):
    return rank(-1 * ((1 - (open_ / close)) ** 1))

def alpha_34(close, returns):
    rank_stddev_returns_2 = rank(stddev(returns, 2))
    rank_stddev_returns_5 = rank(stddev(returns, 5))
    rank_delta_close = rank(delta(close, 1))
    alpha = 1 - rank_stddev_returns_2 / rank_stddev_returns_5 + (1 - rank_delta_close)
    return alpha

def alpha_35(volume, close, high, low, returns):
    rank_volume = rank(volume)
    rank_close_high_low = rank(close + high - low)
    rank_returns = rank(returns)
    alpha = (rank_volume * (1 - rank_close_high_low / 16)) * (1 - rank_returns / 32)
    return alpha

def alpha_36(close, open, volume, vwap, adv20, returns):
    rank_correlation = lambda x, y, d: rank(correlation(x, y, d))
    rank_returns = lambda x, d: rank(ts_rank(x, d))
    rank_abs_correlation = lambda x, y, d: rank(np.abs(correlation(x, y, d)))
    sum_close = lambda d: np.sum(close[-(d-1):]) / d
    rank_diff = lambda x: rank((x[0] - x[1]))

    part1 = 2.21 * rank_correlation(close - open, delta(volume, 1), 15)
    part2 = 0.7 * rank_diff([open, close])
    part3 = 0.73 * rank_returns(delta(-1 * returns, 6), 5)
    part4 = rank_abs_correlation(vwap, adv20, 6)
    part5 = 0.6 * rank_diff([sum_close(200), open]) * rank_diff([close, open])

    return part1 + part2 + part3 + part4 + part5

def alpha_37(close, open):
    return rank(correlation(np.roll((open - close),1), close, 200)) + rank((open - close))

def alpha_38(close, open):
    return (-1 * rank(ts_rank(close, 10))) * rank((close / open))

def alpha_39(close, volume, adv20, returns):
    return (-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(np.sum(returns[-249:])))

def alpha_40(high, volume):
    return (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)

def alpha_41(high, low, vwap):
    return ((high * low) ** 0.5) - vwap

def alpha_42(vwap, close):
    return rank((vwap - close)) / rank((vwap + close))

def alpha_43(volume, adv20, close):
    volume_adv = volume / adv20
    delta_close = delta(close, 7)
    return ts_rank(volume_adv, 20) * ts_rank(-delta_close, 8)

def alpha_44(high, volume):
    return -correlation(high, rank(volume), 5)

def alpha_45(close, volume):
    rank_sum_delay_close_5_20 = rank(np.sum(np.roll(close,5)[-19:]) / 20)
    corr_close_volume = correlation(close, volume, 2)
    rank_corr_close_volume = rank(corr_close_volume)
    sum_close_5 = np.sum(close[-4:])
    sum_close_20 = np.sum(close[-19:])
    corr_sum_close_5_sum_close_20 = correlation(sum_close_5, sum_close_20, 2)
    rank_corr_sum_close_5_sum_close_20 = rank(corr_sum_close_5_sum_close_20)
    return -1 * (rank_sum_delay_close_5_20 * corr_close_volume * rank_corr_sum_close_5_sum_close_20)

def alpha_46(close):
    delay_close_20 = np.roll(close, 20)
    delay_close_10 = np.roll(close, 10)
    delay_close_1 = np.roll(close, 1)
    ratio1 = (delay_close_20 - delay_close_10) / 10
    ratio2 = (delay_close_10 - close) / 10
    ratio_diff = ratio1 - ratio2
    if 0.25 < ratio_diff:
        return -1
    elif ratio_diff < 0:
        return 1
    else:
        return -1 * (close - delay_close_1)

def alpha_47(volume, high, close, vwap):
    rank_inv_close = rank(1 / close)
    prod1 = rank_inv_close * volume
    prod2 = high * rank(high - close)
    sum_high_5 = np.sum(high[-4:])
    prod3 = prod2 / (sum_high_5 / 5)
    prod4 = prod1 * prod3
    delay_vwap = np.roll(vwap, 5)
    return prod4 - rank(vwap - delay_vwap)

def alpha_48(close, indclass):
    delta_close_1 = delta(close, 1)
    delta_delay_close_1 = delta(np.roll(close, 1), 1)
    corr_delta_close_1_delta_delay_close_1 = correlation(delta_close_1, delta_delay_close_1, 250)
    prod = corr_delta_close_1_delta_delay_close_1 * delta_close_1 / close
    indneutralize_prod = indneutralize(prod, indclass.subindustry)
    sum_pow = np.sum(((delta_close_1 / np.roll(close, 1))**2)[-249:])
    return indneutralize_prod / sum_pow

def alpha_49(close):
    delay_close_20 = np.roll(close, 20)
    delay_close_10 = np.roll(close, 10)
    delay_close_1 = np.roll(close, 1)
    ratio = (delay_close_20 - delay_close_10) / 10 - (delay_close_10 - close) / 10
    if ratio < -0.1:
        return 1
    else:
        return -1 * (close - delay_close_1)

def alpha_50(volume, vwap):
    rank_volume = rank(volume)
    rank_vwap = rank(vwap)
    corr_rank_volume_rank_vwap = correlation(rank_volume, rank_vwap, 5)
    rank_corr_rank_volume_rank_vwap = rank(corr_rank_volume_rank_vwap)
    max_rank_corr_rank_volume_rank_vwap = ts_max(rank_corr_rank_volume_rank_vwap, 5)
    return -1 * max_rank_corr_rank_volume_rank_vwap

def alpha_51(close):
    delay_close_20 = np.roll(close, 20)
    delay_close_10 = np.roll(close, 10)
    ratio = ((delay_close_20 - delay_close_10) / 10) - ((delay_close_10 - close) / 10)
    if ratio < -0.05:
        return 1
    else:
        return -1 * (close - np.roll(close, 1))

def alpha_52(low, returns, volume):
    ts_min_low_5 = ts_min(low, 5)
    delay_ts_min_low_5 = np.roll(ts_min_low_5, 5)
    sum_returns_240 = np.sum(returns[-239:])
    sum_returns_20 = np.sum(returns[-19:])
    avg_sum_returns = (sum_returns_240 - sum_returns_20) / 220
    rank_avg_sum_returns = rank(avg_sum_returns)
    ts_rank_volume_5 = ts_rank(volume, 5)
    return -1 * ((((-1 * ts_min_low_5) + delay_ts_min_low_5) * rank_avg_sum_returns) * ts_rank_volume_5)

def alpha_53(close, low, high):
    delta_high_low = ts_max(high, 9) - ts_min(low, 9)
    return -1 * (((close - low) - (high - close)) / delta_high_low)

def alpha_54(close, low, high, open):
    delta_high_low = high - low
    return -1 * (((low - close) * (open**5)) / (delta_high_low * (close**5)))

def alpha_55(high, low, close, volume):
    high_low_12 = (ts_max(high, 12) - ts_min(low, 12))
    rank_data = (close - ts_min(low, 12)) / high_low_12
    corr_rank_data_high_low_rank_volume = correlation(rank(rank_data), rank(volume), 6)
    alpha_55 = -1 * corr_rank_data_high_low_rank_volume
    return alpha_55

def alpha_56(returns, cap):
    sum_returns_10 = np.sum(returns[-9:])
    sum_returns_2 = np.sum(returns[-1:])
    sum_returns_3 = np.sum(returns[-2:])
    alpha_56 = 0 - (1 * (rank(sum_returns_10 / (sum_returns_2 + sum_returns_3)) * rank(returns * cap)))
    return alpha_56

def alpha_57(close, vwap):
    return 0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))

def alpha_60(close, low, high, volume):
    return 0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10)))))

def alpha_61(vwap, adv180):
    return rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282))

def alpha_83(high, low, close, vwap, volume):
    delayed_hl_avg_close = np.roll((high - low) / (np.sum(close[-4:])/5), 2)
    numerator = rank(delayed_hl_avg_close) * rank(rank(volume))
    denominator = (high - low) / (np.sum(close[-4:])/5) / (vwap - close)
    return numerator / denominator

def alpha_101(close, open, high, low):
    return (close - open) / ((high - low) + 0.001)
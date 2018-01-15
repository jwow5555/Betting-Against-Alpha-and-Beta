import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm


########################################## Function Definitions ###############################################
# Rebalance values based on a number of periods
# values could be portfolio weights, beta, weighted market beta, etc...
def rebalance(values, periods):
    rebal = values.iloc[::periods]
    if rebal.index[-1] == '2015-12':
        w = rebal.iloc[np.repeat(np.arange(len(rebal)-1), periods)]
        w = w.append(values.iloc[-1:])
    else:
        w = rebal.iloc[np.repeat(np.arange(len(rebal)), periods)]
        w = w.ix[0:beta_CAPM.shape[0]]
    w.index = beta_CAPM.index
    return w


########################################## Data Processing ###############################################
# Input Riskfree Rate
rf = pd.DataFrame.from_csv('Riskfree.csv')
rf.index = rf.index.strftime('%Y-%m')
# Input Raw Data
df = pd.DataFrame.from_csv('Stockdata.csv')
# Extract date and change to month
df['date'] = pd.to_datetime(df.date)
df['Month'] = df['date'].dt.strftime('%Y-%m')
df = df.rename(columns={'date': 'Date', 'COMNAM': 'Company Name', 'RET': 'Ret', 'vwretd': 'Market_Ret'})
# Extract stock return data
stockret = pd.DataFrame(df, columns=['Month', 'CUSIP', 'Ret']).dropna()
# Convert return data from strings to float
stockret['Ret'] = stockret.Ret.convert_objects(convert_numeric=True)
# Group by company name and date
stockret = stockret.groupby(['CUSIP', 'Month']).Ret.first().unstack().transpose()
# Keep company that has more than 5 years of data
stockret = stockret.loc[:, pd.notnull(stockret).sum() > 60]
# Extract value-weighted market return and order by month
marketret = pd.DataFrame(df, columns=['Month', 'Market_Ret']).dropna().drop_duplicates().sort_values(by='Month')
marketret = marketret.set_index('Month')
# Export data input
with open('returndata.pkl', 'w') as f:
    pickle.dump([rf, df, stockret, marketret], f)

# Getting back the objects:
# with open('returndata.pkl') as f:
    # rf, df, stockret, marketret = pickle.load(f)


########################################## Alpha, Beta Estimation ###############################################
FF = pd.DataFrame.from_csv('FF.csv')
FF.index = FF.index.strftime('%Y-%m')
FF = FF.ix[stockret.index]
excess_ret = np.subtract(stockret, rf)
market_excess = np.subtract(marketret, rf)
FF.to_csv('FF5.csv')
excess_ret.to_csv('excess_ret.csv')
market_excess.to_csv('market_excess.csv')
# I have to switch to Rstudio to perform regression, which is more efficient.
# alpha_CAPM = pd.DataFrame(index=stockret.index, columns=stockret.columns)
# beta_CAPM = pd.DataFrame(index=stockret.index, columns=stockret.columns)
# alpha_FF5 = pd.DataFrame(index=stockret.index, columns=stockret.columns)
# alpha_Car = pd.DataFrame(index=stockret.index, columns=stockret.columns)
#
# for company in excess_ret:
#    for i in range(0, stockret.shape[0]-11):
#        Y = excess_ret.ix[i:(i+12), company]
#        while Y.isna().sum() == 0:
#            X_CAPM = sm.add_constant(market_excess.ix[i:(i+12)])
#            model_CAPM = sm.OLS(Y, X_CAPM)
#            results = model_CAPM.fit()
#            alpha_CAPM.ix[i+11, company] = results.params[0]
#            beta_CAPM.ix[i+11, company] = results.params[1]
#            X_FF5 = sm.add_constant(FF.ix[i:(i+12)].drop(['RF', 'MOM'], axis=1))
#            model_FF5 = sm.OLS(Y, X_FF5)
#            results = model_FF5.fit()
#            alpha_FF5.ix[i+11, company] = results.params[0]
#            X_Car = sm.add_constant(FF.ix[i:(i+12)].drop(['RF'], axis=1))
#            model_Car = sm.OLS(Y, X_Car)
#            results = model_Car.fit()
#            alpha_Car.ix[i+11, company] = results.params[0]

alpha_CAPM = pd.DataFrame.from_csv('alpha_CAPM.csv')
# beta_CAPM = pd.DataFrame.from_csv('beta_CAPM.csv')
alpha_FF5 = pd.DataFrame.from_csv('alpha_FF5.csv')
alpha_Car = pd.DataFrame.from_csv('alpha_Car.csv')
alpha_CAPM = alpha_CAPM.set_index('Month')
# beta_CAPM = beta_CAPM.set_index('Month')
alpha_FF5 = alpha_FF5.set_index('Month')
alpha_Car = alpha_Car.set_index('Month')
alpha_CAPM = alpha_CAPM.rename(columns = lambda x : str(x)[1:])
# beta_CAPM =beta_CAPM.rename(columns = lambda x : str(x)[1:])
alpha_FF5 = alpha_FF5.rename(columns = lambda x : str(x)[1:])
alpha_Car = alpha_Car.rename(columns = lambda x : str(x)[1:])


########################################## Data Statistics ###############################################
# Market Volatility using 1-year data
sigma_m = marketret.rolling(window=12).std().rename(columns={'Market_Ret': 'Market_Vol'})
# Stock Volatility using 1-year data
sigma = stockret.rolling(window=12).std()
# Correlation between stockret and marketret using 5-year data
corr = pd.DataFrame(index=stockret.index, columns=stockret.columns)
for column in stockret:
    corr[column] = pd.rolling_corr(stockret[column], marketret, window=60)

# Export intermediate calculations
with open('calc.pkl', 'w') as f:
    pickle.dump([sigma_m, sigma], f)
# Getting back the objects:
# with open('calc.pkl') as f:
    # sigma_m, sigma= pickle.load(f)

# Delete company that has no data at all times and delete times when no company has data
corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')

alpha_CAPM = alpha_CAPM.dropna(axis=0, how='all').dropna(axis=1, how='all')
# beta_CAPM = beta_CAPM.dropna(axis=0, how='all').dropna(axis=1, how='all')
alpha_FF5 = alpha_FF5.dropna(axis=0, how='all').dropna(axis=1, how='all')
alpha_Car = alpha_Car.dropna(axis=0, how='all').dropna(axis=1, how='all')

# Export intermediate calculations
with open('alphabeta.pkl', 'w') as f:
    pickle.dump([alpha_CAPM, alpha_FF5, alpha_Car], f)
# Getting back the objects:
# with open('alphabeta.pkl') as f:
    # alpha_CAPM, beta_CAPM, alpha_FF5, alpha_Car= pickle.load(f)

marketret = marketret.ix[corr.index]
rf = rf.ix[corr.index]
sigma_m = sigma_m.ix[corr.index]
sigma = sigma.ix[corr.index]
stockret = stockret.ix[corr.index]
alpha_CAPM = alpha_CAPM.ix[corr.index]
alpha_FF5 = alpha_FF5.ix[corr.index]
alpha_Car = alpha_Car.ix[corr.index]

with open('reduced.pkl', 'w') as f:
    pickle.dump([marketret, rf, sigma_m, sigma, stockret], f)
# Getting back the objects:
# with open('reduced.pkl') as f:
    # marketret, rf, sigma_m, sigma, stockret= pickle.load(f)

# Calculate beta
beta_CAPM = np.multiply(corr, np.divide(sigma, sigma_m))
beta_CAPM = np.add(np.multiply(0.6, beta_CAPM), 0.4)
# Calculate CAPM alpha
# alpha = np.subtract(np.subtract(stockret, rf), np.multiply(beta, np.subtract(marketret, rf)))
# Export intermediate calculations
# with open('beta.pkl', 'w') as f:
#    pickle.dump([corr, sigma, sigma_m, marketret, rf, stockret, beta, alpha], f)
# Getting back the objects:
# with open('beta.pkl') as f:
    # corr, sigma, sigma_m, marketret, rf, stockret, beta, alpha= pickle.load(f)


########################################## Portfolio Weights BAB ###############################################
# Calculate median beta at each time period
beta_median = np.nanmedian(beta_CAPM, axis=1)
# Convert array to dataframe and add date as index
beta_median = pd.DataFrame(beta_median, index=beta_CAPM.index).rename(columns={0: 'Beta_Mid'})

# Total number of assets at each month
total = np.subtract(beta_CAPM.shape[1], beta_CAPM.isnull().sum(axis=1))
# Determine if in the high or low beta portfolio
high_BAB = pd.DataFrame(index=beta_CAPM.index, columns=beta_CAPM.columns)
low_BAB = pd.DataFrame(index=beta_CAPM.index, columns=beta_CAPM.columns)
for company in beta_CAPM:
    high_BAB[company] = np.greater(beta_CAPM[company], beta_median.ix[:, 0])
    low_BAB[company] = np.less_equal(beta_CAPM[company], beta_median.ix[:, 0])
# Check
# check = np.count_nonzero(low_BAB, axis=1) + np.count_nonzero(high_BAB, axis=1)
# np.equal(total, check).sum()

# Export high low indicators
with open('highlow_BAB.pkl', 'w') as f:
    pickle.dump([beta_median, high_BAB, low_BAB], f)
# Getting back the objects:
# with open('highlow_BAB.pkl') as f:
    # beta_median, high_BAB, low_BAB = pickle.load(f)

# Number of assets in the low beta portfolio
nl_BAB = low_BAB.sum(axis=1)

# Number of assets in the high beta portfolio
nh_BAB = high_BAB.sum(axis=1)

# Rank beta in low beta portfolio
zl_BAB = beta_CAPM[low_BAB].rank(axis=1)
wl_BAB = np.add(np.absolute(zl_BAB.subtract(nl_BAB, axis='index')), 1).divide(zl_BAB.sum(axis=1), axis='index')
# Check total weight is 1
wl_BAB.sum(axis=1)
# Rank beta in high beta portfolio
zh_BAB = beta_CAPM[high_BAB].rank(axis=1)
wh_BAB = zh_BAB.divide(zh_BAB.sum(axis=1), axis='index')
# Check total weight is 1
wh_BAB.sum(axis=1)

# Final weighted market beta of the low beta portfolio
beta_L = np.multiply(wl_BAB, beta_CAPM[low_BAB]).sum(axis=1)
beta_L = np.add(np.multiply(0.6, beta_L), 0.4)
# Average investment in the long portfolio using yearly rebalanced strategies (end of December)
np.divide(1, beta_L[beta_L.index.str.contains("-12")].mean())
# 1.6096777824851367

# Final weighted market beta of the high beta portfolio
beta_H = np.multiply(wh_BAB, beta_CAPM[high_BAB]).sum(axis=1)
beta_H = np.add(np.multiply(0.6, beta_H), 0.4)
#  Average investment in the short portfolio using yearly rebalanced strategies (end of December)
np.divide(1, beta_H[beta_H.index.str.contains("-12")].mean())
# 0.61266603617578963


########################################## Portfolio Weights BAA ###############################################
# Calculate median alpha at each time period
alpha_median = np.nanmedian(alpha_CAPM, axis=1)
# Convert array to dataframe and add date as index
alpha_median = pd.DataFrame(alpha_median, index=alpha_CAPM.index).rename(columns={0: 'Alpha_Mid'})

# Determine if in the high or low alpha portfolio
high_BAA = pd.DataFrame(index=alpha_CAPM.index, columns=alpha_CAPM.columns)
low_BAA = pd.DataFrame(index=alpha_CAPM.index, columns=alpha_CAPM.columns)
for company in alpha_CAPM:
    high_BAA[company] = np.greater(alpha_CAPM[company], alpha_median.ix[:, 0])
    low_BAA[company] = np.less_equal(alpha_CAPM[company], alpha_median.ix[:, 0])
# Check
# check = np.count_nonzero(low_BAA, axis=1) + np.count_nonzero(high_BAA, axis=1)
# np.equal(total, check).sum()

# Export high low indicators
with open('highlow_BAA.pkl', 'w') as f:
    pickle.dump([alpha_median, high_BAA, low_BAA], f)
# Getting back the objects:
# with open('highlow_BAA.pkl') as f:
    # alpha_median, high_BAA, low_BAA = pickle.load(f)

# Number of assets in the low alpha portfolio
nl_BAA = low_BAA.sum(axis=1)

# Number of assets in the high alpha portfolio
nh_BAA = high_BAA.sum(axis=1)

# Rank alpha in low alpha portfolio
zl_BAA = alpha_CAPM[low_BAA].rank(axis=1)
wl_BAA = np.add(np.absolute(zl_BAA.subtract(nl_BAA, axis='index')), 1).divide(zl_BAA.sum(axis=1), axis='index')
# Check total weight is 1
wl_BAA.sum(axis=1)
# Rank alpha in high alpha portfolio
zh_BAA = alpha_CAPM[high_BAA].rank(axis=1)
wh_BAA = zh_BAA.divide(zh_BAA.sum(axis=1), axis='index')
# Check total weight is 1
wh_BAA.sum(axis=1)


########################################## Portfolio Weights BAAB ###############################################
hahb_BAAB = pd.DataFrame(index=alpha_CAPM.index, columns=alpha_CAPM.columns)
halb_BAAB = pd.DataFrame(index=alpha_CAPM.index, columns=alpha_CAPM.columns)
lahb_BAAB = pd.DataFrame(index=alpha_CAPM.index, columns=alpha_CAPM.columns)
lalb_BAAB = pd.DataFrame(index=alpha_CAPM.index, columns=alpha_CAPM.columns)
for company in alpha_CAPM:
    hahb_BAAB[company] = np.greater(alpha_CAPM[high_BAB][company], alpha_median.ix[:, 0])
    lahb_BAAB[company] = np.less_equal(alpha_CAPM[high_BAB][company], alpha_median.ix[:, 0])
    halb_BAAB[company] = np.greater(alpha_CAPM[low_BAB][company], alpha_median.ix[:, 0])
    lalb_BAAB[company] = np.less_equal(alpha_CAPM[low_BAB][company], alpha_median.ix[:, 0])

## Low beta portfolio
# Number of assets in the low alpha portfolio
nlalb_BAAB = lalb_BAAB.sum(axis=1)

# Number of assets in the high alpha portfolio
nhalb_BAAB = halb_BAAB.sum(axis=1)

# Rank alpha in low alpha portfolio
zlalb_BAAB = alpha_CAPM[lalb_BAAB].rank(axis=1)
wlalb_BAAB = np.add(np.absolute(zlalb_BAAB.subtract(nlalb_BAAB, axis='index')), 1).divide(zlalb_BAAB.sum(axis=1), axis='index')
# Check total weight is 1
wlalb_BAAB.sum(axis=1)
# Rank alpha in high alpha portfolio
zhalb_BAAB = alpha_CAPM[halb_BAAB].rank(axis=1)
whalb_BAAB = zhalb_BAAB.divide(zhalb_BAAB.sum(axis=1), axis='index')
# Check total weight is 1
whalb_BAAB.sum(axis=1)

## High beta portfolio
# Number of assets in the low alpha portfolio
nlahb_BAAB = lahb_BAAB.sum(axis=1)

# Number of assets in the high alpha portfolio
nhahb_BAAB = hahb_BAAB.sum(axis=1)

# Rank alpha in low alpha portfolio
zlahb_BAAB = alpha_CAPM[lahb_BAAB].rank(axis=1)
wlahb_BAAB = np.add(np.absolute(zlahb_BAAB.subtract(nlahb_BAAB, axis='index')), 1).divide(zlahb_BAAB.sum(axis=1), axis='index')
# Check total weight is 1
wlahb_BAAB.sum(axis=1)
# Rank alpha in high alpha portfolio
zhahb_BAAB = alpha_CAPM[hahb_BAAB].rank(axis=1)
whahb_BAAB = zhahb_BAAB.divide(zhahb_BAAB.sum(axis=1), axis='index')
# Check total weight is 1
whahb_BAAB.sum(axis=1)


########################################## Empirical Tests ###############################################

########################################## Figure 1 ###############################################
plt.close('all')
fig, ax = plt.subplots(1)
ax.plot(np.divide(1, beta_L[beta_L.index.str.contains("-12")]), 'b-', label="Low beta portfolio weight")
ax.plot(np.divide(1, beta_H[beta_H.index.str.contains("-12")]), 'r-', label="High beta portfolio weight")
fig.autofmt_xdate()
plt.title('High beta and low beta portfolios weights at the end of every December')
plt.legend()
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)


########################################## Table 1 ###############################################
# The holding period return is 12 months
# Betting Against Beta
# Excess Return
# Low Beta
wl_BAB_12 = rebalance(wl_BAB, 48)
excess_BAB_lb_12 = np.subtract(np.multiply(wl_BAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAB_lb_12.mean()

alpha_BAB_lb_12_CAPM = np.multiply(wl_BAB_12, alpha_CAPM).sum(axis=1)
alpha_BAB_lb_12_FF5 = np.multiply(wl_BAB_12, alpha_FF5).sum(axis=1)
alpha_BAB_lb_12_Car = np.multiply(wl_BAB_12, alpha_Car).sum(axis=1)
alpha_BAB_lb_12_CAPM.mean()

alpha_BAB_lb_12_FF5.mean()

alpha_BAB_lb_12_Car.mean()

totalret_BAB_lb_12 = np.multiply(wl_BAB_12, stockret).sum(axis=1)
sharpe_BAB_lb_12 = np.divide(excess_BAB_lb_12.mean(), np.std(totalret_BAB_lb_12))
sharpe_BAB_lb_12


# High Beta
wh_BAB_12 = rebalance(wh_BAB, 48)
excess_BAB_hb_12 = np.subtract(np.multiply(wh_BAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAB_hb_12.mean()

alpha_BAB_hb_12_CAPM = np.multiply(wh_BAB_12, alpha_CAPM).sum(axis=1)
alpha_BAB_hb_12_FF5 = np.multiply(wh_BAB_12, alpha_FF5).sum(axis=1)
alpha_BAB_hb_12_Car = np.multiply(wh_BAB_12, alpha_Car).sum(axis=1)
alpha_BAB_hb_12_CAPM.mean()

alpha_BAB_hb_12_FF5.mean()

alpha_BAB_hb_12_Car.mean()

totalret_BAB_hb_12 = np.multiply(wh_BAB_12, stockret).sum(axis=1)
sharpe_BAB_hb_12 = np.divide(excess_BAB_hb_12.mean(), np.std(totalret_BAB_hb_12))
sharpe_BAB_hb_12


# Low - High
beta_BAB_lb_12 = rebalance(beta_L, 48)
beta_BAB_hb_12 = rebalance(beta_H, 48)
excess_BAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAB_lb_12), excess_BAB_lb_12),
                               np.multiply(np.divide(1, beta_BAB_hb_12), excess_BAB_hb_12))
excess_BAB_lh_12.mean()

alpha_BAB_lh_12_CAPM = np.subtract(np.multiply(np.divide(1, beta_BAB_lb_12), alpha_BAB_lb_12_CAPM),
                                   np.multiply(np.divide(1, beta_BAB_hb_12), alpha_BAB_hb_12_CAPM))
alpha_BAB_lh_12_FF5 = np.subtract(np.multiply(np.divide(1, beta_BAB_lb_12), alpha_BAB_lb_12_FF5),
                                  np.multiply(np.divide(1, beta_BAB_hb_12), alpha_BAB_hb_12_FF5))
alpha_BAB_lh_12_Car = np.subtract(np.multiply(np.divide(1, beta_BAB_lb_12), alpha_BAB_lb_12_Car),
                                  np.multiply(np.divide(1, beta_BAB_hb_12), alpha_BAB_hb_12_Car))
alpha_BAB_lh_12_CAPM.mean()

alpha_BAB_lh_12_FF5.mean()

alpha_BAB_lh_12_Car.mean()


totalret_BAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAB_lb_12), totalret_BAB_lb_12),
                                 np.multiply(np.divide(1, beta_BAB_hb_12), totalret_BAB_hb_12))
sharpe_BAB_lh_12 = np.divide(excess_BAB_lh_12.mean(), np.std(totalret_BAB_lh_12))
sharpe_BAB_lh_12


##################################################################################################

# Betting Against Alpha
# Excess Return
# Low Alpha
wl_BAA_12 = rebalance(wl_BAA, 48)
excess_BAA_lb_12 = np.subtract(np.multiply(wl_BAA_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAA_lb_12.mean()

alpha_BAA_lb_12_CAPM = np.multiply(wl_BAA_12, alpha_CAPM).sum(axis=1)
alpha_BAA_lb_12_FF5 = np.multiply(wl_BAA_12, alpha_FF5).sum(axis=1)
alpha_BAA_lb_12_Car = np.multiply(wl_BAA_12, alpha_Car).sum(axis=1)
alpha_BAA_lb_12_CAPM.mean()

alpha_BAA_lb_12_FF5.mean()

alpha_BAA_lb_12_Car.mean()

totalret_BAA_lb_12 = np.multiply(wl_BAA_12, stockret).sum(axis=1)
sharpe_BAA_lb_12 = np.divide(excess_BAA_lb_12.mean(), np.std(totalret_BAA_lb_12))
sharpe_BAA_lb_12



# High Alpha
wh_BAA_12 = rebalance(wh_BAA, 48)
excess_BAA_hb_12 = np.subtract(np.multiply(wh_BAA_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAA_hb_12.mean()

alpha_BAA_hb_12_CAPM = np.multiply(wh_BAA_12, alpha_CAPM).sum(axis=1)
alpha_BAA_hb_12_FF5 = np.multiply(wh_BAA_12, alpha_FF5).sum(axis=1)
alpha_BAA_hb_12_Car = np.multiply(wh_BAA_12, alpha_Car).sum(axis=1)
alpha_BAA_hb_12_CAPM.mean()

alpha_BAA_hb_12_FF5.mean()

alpha_BAA_hb_12_Car.mean()

totalret_BAA_hb_12 = np.multiply(wh_BAA_12, stockret).sum(axis=1)
sharpe_BAA_hb_12 = np.divide(excess_BAA_hb_12.mean(), np.std(totalret_BAA_hb_12))
sharpe_BAA_hb_12


# Low - High
beta_BAA_lb_12 = rebalance(beta_L, 48)
beta_BAA_hb_12 = rebalance(beta_H, 48)
excess_BAA_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAA_lb_12), excess_BAA_lb_12),
                               np.multiply(np.divide(1, beta_BAA_hb_12), excess_BAA_hb_12))
excess_BAA_lh_12.mean()

alpha_BAA_lh_12_CAPM = np.subtract(np.multiply(np.divide(1, beta_BAA_lb_12), alpha_BAA_lb_12_CAPM),
                                   np.multiply(np.divide(1, beta_BAA_hb_12), alpha_BAA_hb_12_CAPM))
alpha_BAA_lh_12_FF5 = np.subtract(np.multiply(np.divide(1, beta_BAA_lb_12), alpha_BAA_lb_12_FF5),
                                   np.multiply(np.divide(1, beta_BAA_hb_12), alpha_BAA_hb_12_FF5))
alpha_BAA_lh_12_Car = np.subtract(np.multiply(np.divide(1, beta_BAA_lb_12), alpha_BAA_lb_12_Car),
                                   np.multiply(np.divide(1, beta_BAA_hb_12), alpha_BAA_hb_12_Car))
alpha_BAA_lh_12_CAPM.mean()

alpha_BAA_lh_12_FF5.mean()

alpha_BAA_lh_12_Car.mean()


totalret_BAA_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAA_lb_12), totalret_BAA_lb_12),
                                 np.multiply(np.divide(1, beta_BAA_hb_12), totalret_BAA_hb_12))
sharpe_BAA_lh_12 = np.divide(excess_BAA_lh_12.mean(), np.std(totalret_BAA_lh_12))
sharpe_BAA_lh_12


##################################################################################################

# Betting Against Alpha and Beta
## In the low beta portfolio
# Excess Return
# Low Alpha
wlalb_BAAB_12 = rebalance(wlalb_BAAB, 48)
excess_BAAB_lalb_12 = np.subtract(np.multiply(wlalb_BAAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAAB_lalb_12.mean()

alpha_BAAB_lalb_12_CAPM = np.multiply(wlalb_BAAB_12, alpha_CAPM).sum(axis=1)
alpha_BAAB_lalb_12_FF5 = np.multiply(wlalb_BAAB_12, alpha_FF5).sum(axis=1)
alpha_BAAB_lalb_12_Car = np.multiply(wlalb_BAAB_12, alpha_Car).sum(axis=1)
alpha_BAAB_lalb_12_CAPM.mean()

alpha_BAAB_lalb_12_FF5.mean()

alpha_BAAB_lalb_12_Car.mean()

totalret_BAAB_lalb_12 = np.multiply(wlalb_BAAB_12, stockret).sum(axis=1)
sharpe_BAAB_lalb_12 = np.divide(excess_BAAB_lalb_12.mean(), np.std(totalret_BAAB_lalb_12))
sharpe_BAAB_lalb_12



# High Alpha
whalb_BAAB_12 = rebalance(whalb_BAAB, 48)
excess_BAAB_halb_12 = np.subtract(np.multiply(whalb_BAAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAAB_halb_12.mean()

alpha_BAAB_halb_12_CAPM = np.multiply(whalb_BAAB_12, alpha_CAPM).sum(axis=1)
alpha_BAAB_halb_12_FF5 = np.multiply(whalb_BAAB_12, alpha_FF5).sum(axis=1)
alpha_BAAB_halb_12_Car = np.multiply(whalb_BAAB_12, alpha_Car).sum(axis=1)
alpha_BAAB_halb_12_CAPM.mean()

alpha_BAAB_halb_12_FF5.mean()

alpha_BAAB_halb_12_Car.mean()

totalret_BAAB_halb_12 = np.multiply(whalb_BAAB_12, stockret).sum(axis=1)
sharpe_BAAB_halb_12 = np.divide(excess_BAAB_halb_12.mean(), np.std(totalret_BAAB_halb_12))
sharpe_BAAB_halb_12


# Low - High
beta_BAAB_lb_12 = rebalance(beta_L, 48)
beta_BAAB_hb_12 = rebalance(beta_H, 48)
excess_BAAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), excess_BAAB_lalb_12),
                               np.multiply(np.divide(1, beta_BAAB_hb_12), excess_BAAB_halb_12))
excess_BAAB_lh_12.mean()

alpha_BAAB_lh_12_CAPM = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lalb_12_CAPM),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_halb_12_CAPM))
alpha_BAAB_lh_12_FF5 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lalb_12_FF5),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_halb_12_FF5))
alpha_BAAB_lh_12_Car = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lalb_12_Car),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_halb_12_Car))
alpha_BAAB_lh_12_CAPM.mean()

alpha_BAAB_lh_12_FF5.mean()

alpha_BAAB_lh_12_Car.mean()


totalret_BAAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), totalret_BAAB_lalb_12),
                                 np.multiply(np.divide(1, beta_BAAB_hb_12), totalret_BAAB_halb_12))
sharpe_BAAB_lh_12 = np.divide(excess_BAAB_lh_12.mean(), np.std(totalret_BAAB_lh_12))
sharpe_BAAB_lh_12


##################################################################################################
## In the high beta portfolio
# Excess Return
# Low Alpha
wlahb_BAAB_12 = rebalance(wlahb_BAAB, 48)
excess_BAAB_lahb_12 = np.subtract(np.multiply(wlahb_BAAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAAB_lahb_12.mean()

alpha_BAAB_lahb_12_CAPM = np.multiply(wlahb_BAAB_12, alpha_CAPM).sum(axis=1)
alpha_BAAB_lahb_12_FF5 = np.multiply(wlahb_BAAB_12, alpha_FF5).sum(axis=1)
alpha_BAAB_lahb_12_Car = np.multiply(wlahb_BAAB_12, alpha_Car).sum(axis=1)
alpha_BAAB_lahb_12_CAPM.mean()

alpha_BAAB_lahb_12_FF5.mean()

alpha_BAAB_lahb_12_Car.mean()

totalret_BAAB_lahb_12 = np.multiply(wlahb_BAAB_12, stockret).sum(axis=1)
sharpe_BAAB_lahb_12 = np.divide(excess_BAAB_lahb_12.mean(), np.std(totalret_BAAB_lahb_12))
sharpe_BAAB_lahb_12



# High Alpha
whahb_BAAB_12 = rebalance(whahb_BAAB, 48)
excess_BAAB_hahb_12 = np.subtract(np.multiply(whahb_BAAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAAB_hahb_12.mean()

alpha_BAAB_hahb_12_CAPM = np.multiply(whahb_BAAB_12, alpha_CAPM).sum(axis=1)
alpha_BAAB_hahb_12_FF5 = np.multiply(whahb_BAAB_12, alpha_FF5).sum(axis=1)
alpha_BAAB_hahb_12_Car = np.multiply(whahb_BAAB_12, alpha_Car).sum(axis=1)
alpha_BAAB_hahb_12_CAPM.mean()

alpha_BAAB_hahb_12_FF5.mean()

alpha_BAAB_hahb_12_Car.mean()

totalret_BAAB_hahb_12 = np.multiply(whahb_BAAB_12, stockret).sum(axis=1)
sharpe_BAAB_hahb_12 = np.divide(excess_BAAB_hahb_12.mean(), np.std(totalret_BAAB_hahb_12))
sharpe_BAAB_hahb_12


# Low - High
beta_BAAB_lb_12 = rebalance(beta_L, 48)
beta_BAAB_hb_12 = rebalance(beta_H, 48)
excess_BAAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), excess_BAAB_lahb_12),
                               np.multiply(np.divide(1, beta_BAAB_hb_12), excess_BAAB_hahb_12))
excess_BAAB_lh_12.mean()

alpha_BAAB_lh_12_CAPM = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lahb_12_CAPM),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_hahb_12_CAPM))
alpha_BAAB_lh_12_FF5 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lahb_12_FF5),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_hahb_12_FF5))
alpha_BAAB_lh_12_Car = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lahb_12_Car),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_hahb_12_Car))
alpha_BAAB_lh_12_CAPM.mean()

alpha_BAAB_lh_12_FF5.mean()

alpha_BAAB_lh_12_Car.mean()


totalret_BAAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), totalret_BAAB_lahb_12),
                                 np.multiply(np.divide(1, beta_BAAB_hb_12), totalret_BAAB_hahb_12))
sharpe_BAAB_lh_12 = np.divide(excess_BAAB_lh_12.mean(), np.std(totalret_BAAB_lh_12))
sharpe_BAAB_lh_12



##################################################################################################
### Low Beta & Alpha - High Beta & Alpha
# Excess Return
# Low Alpha
wlalb_BAAB_12 = rebalance(wlalb_BAAB, 48)
excess_BAAB_lalb_12 = np.subtract(np.multiply(wlalb_BAAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAAB_lalb_12.mean()

alpha_BAAB_lalb_12_CAPM = np.multiply(wlalb_BAAB_12, alpha_CAPM).sum(axis=1)
alpha_BAAB_lalb_12_FF5 = np.multiply(wlalb_BAAB_12, alpha_FF5).sum(axis=1)
alpha_BAAB_lalb_12_Car = np.multiply(wlalb_BAAB_12, alpha_Car).sum(axis=1)
alpha_BAAB_lalb_12_CAPM.mean()

alpha_BAAB_lalb_12_FF5.mean()

alpha_BAAB_lalb_12_Car.mean()

totalret_BAAB_lalb_12 = np.multiply(wlalb_BAAB_12, stockret).sum(axis=1)
sharpe_BAAB_lalb_12 = np.divide(excess_BAAB_lalb_12.mean(), np.std(totalret_BAAB_lalb_12))
sharpe_BAAB_lalb_12



# High Alpha
whahb_BAAB_12 = rebalance(whahb_BAAB, 48)
excess_BAAB_hahb_12 = np.subtract(np.multiply(whahb_BAAB_12, stockret).sum(axis=1), rf.ix[:, 0])
excess_BAAB_hahb_12.mean()

alpha_BAAB_hahb_12_CAPM = np.multiply(whahb_BAAB_12, alpha_CAPM).sum(axis=1)
alpha_BAAB_hahb_12_FF5 = np.multiply(whahb_BAAB_12, alpha_FF5).sum(axis=1)
alpha_BAAB_hahb_12_Car = np.multiply(whahb_BAAB_12, alpha_Car).sum(axis=1)
alpha_BAAB_hahb_12_CAPM.mean()

alpha_BAAB_hahb_12_FF5.mean()

alpha_BAAB_hahb_12_Car.mean()

totalret_BAAB_hahb_12 = np.multiply(whahb_BAAB_12, stockret).sum(axis=1)
sharpe_BAAB_hahb_12 = np.divide(excess_BAAB_hahb_12.mean(), np.std(totalret_BAAB_hahb_12))
sharpe_BAAB_hahb_12


# Low - High
beta_BAAB_lb_12 = rebalance(beta_L, 48)
beta_BAAB_hb_12 = rebalance(beta_H, 48)
excess_BAAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), excess_BAAB_lalb_12),
                               np.multiply(np.divide(1, beta_BAAB_hb_12), excess_BAAB_hahb_12))
excess_BAAB_lh_12.mean()

alpha_BAAB_lh_12_CAPM = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lalb_12_CAPM),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_hahb_12_CAPM))
alpha_BAAB_lh_12_FF5 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lalb_12_FF5),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_hahb_12_FF5))
alpha_BAAB_lh_12_Car = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), alpha_BAAB_lalb_12_Car),
                                   np.multiply(np.divide(1, beta_BAAB_hb_12), alpha_BAAB_hahb_12_Car))
alpha_BAAB_lh_12_CAPM.mean()

alpha_BAAB_lh_12_FF5.mean()

alpha_BAAB_lh_12_Car.mean()


totalret_BAAB_lh_12 = np.subtract(np.multiply(np.divide(1, beta_BAAB_lb_12), totalret_BAAB_lalb_12),
                                 np.multiply(np.divide(1, beta_BAAB_hb_12), totalret_BAAB_hahb_12))
sharpe_BAAB_lh_12 = np.divide(excess_BAAB_lh_12.mean(), np.std(totalret_BAAB_lh_12))
sharpe_BAAB_lh_12



##################################################################################################
## Table 4: Correlation between factors
# will be done in R programming
totalret_BAB_lh_12.to_csv('BAB.csv')
totalret_BAA_lh_12.to_csv('BAA.csv')
totalret_BAAB_lh_12.to_csv('BAAB.csv')

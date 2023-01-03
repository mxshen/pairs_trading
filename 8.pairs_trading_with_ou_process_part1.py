#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


bbh =  ['MRNA', 'AMGN', 'IQV', 'ICLR', 'VRTX', 'GILD', 'REGN', 'ILMN', 'CRL', 'SGEN', 'BIIB', 
        'TECH', 'BNTX', 'BGNE', 'EXAS', 'ALNY', 'NVAX', 'QGEN', 'GH', 'NTRA', 'BMRN', 'INCY', 
        'TXG', 'NTLA', 'CRSP']

stocks = bbh
etfs = ['BBH']


# In[3]:


prices = pd.read_csv('bbh_idna.csv', index_col=0)
prices = prices[stocks+etfs]


# In[4]:


returns = prices.pct_change().dropna()


# In[8]:


returns


# In[6]:


returns.index[59]


# In[7]:


s_scores = pd.DataFrame(index=returns.index[59:], columns=stocks)
betas = pd.DataFrame(index=returns.index[59:], columns=stocks)


# In[31]:


for t in returns.index[59:]:
    
    # prepare data
    # 采用60天滑动窗口来进行计算
    tmp_ret = returns.loc[:t]
    tmp_ret = tmp_ret.iloc[-60:]
    # normalization
    tmp_ret = (tmp_ret - tmp_ret.mean()) / tmp_ret.std()
    
    ou_parameters = pd.DataFrame(index=stocks, columns=['a', 'b', 'Var(zeta)', 'kappa', 'm', 
                                                      'sigma', 'sigma_eq'])
    
    for stock in tmp_ret.columns.drop(etfs):
        X = tmp_ret[etfs].values   # etf是x
        y = tmp_ret[stock].values
        model1 = LinearRegression().fit(X,y) 
        betas.loc[t,stock] = model1.coef_
        epsilon = y - model1.predict(X)  # Xt

        # second regression
        Xk = epsilon.cumsum()
        
        # 残差的累积 错位回归
        X = Xk[:-1].reshape(-1,1)
        y = Xk[1:]
        model2 = LinearRegression().fit(X,y)
        
        a = model2.intercept_
        b = model2.coef_
        
        # 第二次回归的残差
        zeta = y - model2.predict(X)

        # OU parameters
        kappa = -np.log(b)*252
        m = a/(1-b)
        sigma = np.sqrt(np.var(zeta)*2*kappa/(1-b**2))
        sigma_eq = np.sqrt(np.var(zeta)/(1-b**2))

        # if the speed of mean reversion is high enough, save the calculated parameters
        if kappa>252/30:
            ou_parameters.loc[stock] = [x.item() for x in [a,b,np.var(zeta),kappa,m,sigma,sigma_eq]]

    ou_parameters.dropna(axis=0, inplace=True)

    # calculate s-score
    ou_parameters['m_bar'] = (ou_parameters['a']/(1 - ou_parameters['b']) - 
                              ou_parameters['a'].mean()/(1-ou_parameters['b'].mean()))
    ou_parameters['s'] = -ou_parameters['m_bar'] / ou_parameters['sigma_eq']
    s_scores.loc[t] = ou_parameters['s']


# In[32]:


s_scores


# In[33]:

s_scores['MRNA'].plot(figsize=(18,6))


# In[34]:


betas  # 买多少手 etf


# In[35]:


# calculate positions
algo_pos = pd.DataFrame(index=s_scores.index[1:], columns=stocks)

for s in stocks:
    positions = pd.DataFrame(index=s_scores.index, columns=[s])
    pos = 0
    for t in s_scores.index:
        score = s_scores.loc[t][s]
        if score>1.25:
            positions.loc[t][s] = -1 # open short
            pos = -1
        elif score<-1.25:
            positions.loc[t][s] = 1 # open long
            pos = 1
        elif score<0.75 and pos==-1: 
            positions.loc[t][s] = 0 # close short
            pos = 0
        elif score>-0.5 and pos==1:
            positions.loc[t][s] = 0 # close long
            pos = 0
        else:
            positions.loc[t][s] = pos # carry forward current position

    algo_pos[s] = positions


# In[36]:


algo_pos


# In[37]:


# calculate weights (allocate equal amount of capital to long\short positions)
algo_weights = pd.DataFrame(index=algo_pos.index, columns=stocks)

for t in algo_pos.index:
    tmp = algo_pos.loc[t]
    tmp[tmp>0] /= sum(tmp>0) # equal weights among long positions
    tmp[tmp<0] /= sum(tmp<0) # equal weights among short positions
    algo_weights.loc[t] = tmp


# In[38]:


algo_weights


# In[39]:


# calculate positions in ETFs
algo_weights[etfs] = -(betas.iloc[1:,:]*algo_weights).sum(axis=1).values.reshape(-1,1)


# In[ ]:


algo_weights[etfs]


# In[ ]:


# calculate returns
ret = (returns.iloc[60:] * algo_weights.shift()).sum(axis=1) / (abs(algo_weights.shift()).sum(axis=1)/2)
cumret = np.nancumprod(ret+1)

plt.plot(cumret)
# In[40]:


# calculate returns of SPY and BBH for comparison
spy = pd.read_csv('spy_ohlc.csv', index_col=0)
spy_returns = spy['5. adjusted close'].pct_change()
spy_returns = spy_returns.loc[returns.index]
spy_cumret = np.nancumprod(spy_returns.iloc[60:]+1)
bbh_cumret = np.nancumprod(returns.iloc[60:]['BBH']+1)

plt.plot(spy_cumret)
plt.plot(bbh_cumret)

# In[41]:


plt.figure(figsize=(18,6))
plt.plot(cumret, label='Algo')
plt.plot(spy_cumret, label='SPY')
plt.plot(bbh_cumret, label='BBH')
plt.legend()


# In[ ]:


# fraction of returns to pay transaction costs for
tc_frac = abs(algo_pos.shift().diff()).sum(axis=1)/abs(algo_pos.shift()).sum(axis=1)
# assume two-way transaction cost of 0.1%
ret_tc = ret - 0.0005*2*tc_frac # multiply by 2 since we use 2x capital
cumret_tc = np.nancumprod(1+ret_tc)


# In[ ]:


plt.figure(figsize=(18,6))
plt.plot(cumret_tc, label='Algo with tc')
plt.plot(spy_cumret, label='SPY')
plt.plot(bbh_cumret, label='BBH')
plt.legend()


# In[ ]:


def calculate_metrics(cumret):
    '''
    calculate performance metrics from cumulative returns
    '''
    total_return = (cumret[-1] - cumret[0])/cumret[0]
    apr = (1+total_return)**(252/len(cumret)) - 1
    rets = pd.DataFrame(cumret).pct_change()
    sharpe = np.sqrt(252) * np.nanmean(rets) / np.nanstd(rets)
    
    # maxdd and maxddd
    highwatermark=np.zeros(cumret.shape)
    drawdown=np.zeros(cumret.shape)
    drawdownduration=np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t]=cumret[t]/highwatermark[t]-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1
    maxDD=np.min(drawdown)
    maxDDD=np.max(drawdownduration)
    
    return total_return, apr, sharpe, maxDD, maxDDD


# In[ ]:


results = pd.DataFrame(index=['SPY', 'BBH', 'Algo', 'Algo_with_tc'], columns=['total_return', 'apr', 'Sharpe', 
                                                                       'maxDD', 'maxDDD'])
results.loc['SPY'] = calculate_metrics(spy_cumret)
results.loc['BBH'] = calculate_metrics(bbh_cumret)
results.loc['Algo'] = calculate_metrics(cumret)
results.loc['Algo_with_tc'] = calculate_metrics(cumret_tc)
results


# In[ ]:


total_returns = []

for _ in range(10000):
    p = np.array(5*[0.2] + 4*[-0.25] + 17*[0])
    mc_weights = []

    for i in range(len(algo_pos)):
        mc_weights.append(np.random.permutation(p))
    
    mc_weights = pd.DataFrame(mc_weights, index=algo_weights.index, columns=algo_weights.columns)
    
    # calculate returns
    mc_ret = (returns.iloc[60:] * mc_weights.shift()).sum(axis=1) / (abs(mc_weights.shift()).sum(axis=1)/2)
    mc_cumret = np.nancumprod(mc_ret+1)
    tot_ret = (mc_cumret[-1] - mc_cumret[0])/mc_cumret[0]
    total_returns.append(tot_ret)


# In[ ]:


total_returns = np.array(total_returns)
(total_returns>1.56).sum()/10000


# # XLF ETF and its constituents

# In[ ]:


xlf =  ['BRK.B', 'JPM', 'BAC', 'WFC', 'MS', 'C', 'GS', 'BLK', 'SCHW', 'SPGI', 'AXP', 'PNC', 'MMC', 'TFC', 'CB']
stocks = list(set(xlf))
etfs = ['XLF']
symbols = stocks + etfs
prices = pd.read_csv('xlf.csv', index_col=0)
prices = prices[stocks+etfs]
returns = prices.pct_change().dropna()
returns = returns.loc['2019-10-11':]


# In[ ]:


s_scores = pd.DataFrame(index=returns.index[59:], columns=stocks)
betas = pd.DataFrame(index=returns.index[59:], columns=stocks)

for t in returns.index[59:]:
    # prepare data
    tmp_ret = returns.loc[:t]
    tmp_ret = tmp_ret.iloc[-60:]
    tmp_ret = (tmp_ret - tmp_ret.mean()) / tmp_ret.std()
    
    ou_parameters = pd.DataFrame(index=stocks, columns=['a', 'b', 'Var(zeta)', 'kappa', 'm', 
                                                      'sigma', 'sigma_eq'])
    
    for stock in tmp_ret.columns.drop(etfs):
        X = tmp_ret[etfs].values
        y = tmp_ret[stock].values
        model1 = LinearRegression().fit(X,y) 
        betas.loc[t,stock] = model1.coef_
        epsilon = y - model1.predict(X)

        # second regression
        Xk = epsilon.cumsum()
        X = Xk[:-1].reshape(-1,1)
        y = Xk[1:]
        model2 = LinearRegression().fit(X,y)
        a = model2.intercept_
        b = model2.coef_
        zeta = y - model2.predict(X)

        # OU parameters
        kappa = -np.log(b)*252
        m = a/(1-b)
        sigma = np.sqrt(np.var(zeta)*2*kappa/(1-b**2))
        sigma_eq = np.sqrt(np.var(zeta)/(1-b**2))

        # if the speed of mean reversion is high enough, save the calculated parameters
        if kappa>252/30:
            ou_parameters.loc[stock] = [x.item() for x in [a,b,np.var(zeta),kappa,m,sigma,sigma_eq]]

    ou_parameters.dropna(axis=0, inplace=True)

    # calculate s-score
    ou_parameters['m_bar'] = (ou_parameters['a']/(1 - ou_parameters['b']) - 
                              ou_parameters['a'].mean()/(1-ou_parameters['b'].mean()))
    ou_parameters['s'] = -ou_parameters['m_bar'] / ou_parameters['sigma_eq']
    s_scores.loc[t] = ou_parameters['s']


# In[ ]:


# calculate positions
algo_pos = pd.DataFrame(index=s_scores.index[1:], columns=stocks)

for s in stocks:
    positions = pd.DataFrame(index=s_scores.index, columns=[s])
    pos = 0
    for t in s_scores.index:
        score = s_scores.loc[t][s]
        if score>1.25:
            positions.loc[t][s] = -1 # open short
            pos = -1
        elif score<-1.25:
            positions.loc[t][s] = 1 # open long
            pos = 1
        elif score<0.75 and pos==-1: 
            positions.loc[t][s] = 0 # close short
            pos = 0
        elif score>-0.5 and pos==1:
            positions.loc[t][s] = 0 # close long
            pos = 0
        else:
            positions.loc[t][s] = pos # carry forward current position

    algo_pos[s] = positions
    
# calculate weights (allocate equal amount of capital to long\short positions)
algo_weights = pd.DataFrame(index=algo_pos.index, columns=stocks)

for t in algo_pos.index:
    tmp = algo_pos.loc[t]
    tmp[tmp>0] /= sum(tmp>0) # equal weights among long positions
    tmp[tmp<0] /= sum(tmp<0) # equal weights among short positions
    algo_weights.loc[t] = tmp
    
# calculate positions in ETFs
algo_weights[etfs] = -np.stack((betas.iloc[1:,:]*algo_weights).sum(axis=1).values)

# calculate returns
a1 = returns.iloc[60:]
a2 = algo_weights.shift()


ret = (returns.iloc[60:] * algo_weights.shift()).sum(axis=1) / (abs(algo_weights.shift()).sum(axis=1)/2)
cumret = np.nancumprod(ret+1)

# XLF returns
xlf_cumret = np.nancumprod(returns.iloc[60:]['XLF']+1)


# In[ ]:


plt.figure(figsize=(18,6))
plt.plot(cumret, label='Algo')
plt.plot(spy_cumret, label='SPY')
plt.plot(xlf_cumret, label='XLF')
plt.legend()


# In[ ]:


# fraction of returns to pay transaction costs for
tc_frac = abs(algo_pos.shift().diff()).sum(axis=1)/abs(algo_pos.shift()).sum(axis=1)
# assume two-way transaction cost of 0.1%
ret_tc = ret - 0.0005*2*tc_frac # multiply by 2 since we use 2x capital
cumret_tc = np.nancumprod(1+ret_tc)


# In[ ]:


plt.figure(figsize=(18,6))
plt.plot(cumret_tc, label='Algo with tc')
plt.plot(spy_cumret, label='SPY')
plt.plot(xlf_cumret, label='XLF')
plt.legend()


# In[ ]:


results = pd.DataFrame(index=['SPY', 'XLF', 'Algo', 'Algo_with_tc'], columns=['total_return', 'apr', 'Sharpe', 
                                                                       'maxDD', 'maxDDD'])
results.loc['SPY'] = calculate_metrics(spy_cumret)
results.loc['XLF'] = calculate_metrics(xlf_cumret)
results.loc['Algo'] = calculate_metrics(cumret)
results.loc['Algo_with_tc'] = calculate_metrics(cumret_tc)
results


# In[ ]:


# avg number of long positions
(algo_weights>0).sum(axis=1).mean()


# In[ ]:


# avg number of short positions
(algo_weights<0).sum(axis=1).mean()


# In[ ]:


total_returns = []

for _ in range(10000):
    p = np.array(3*[0.333] + 3*[-0.333] + 10*[0])
    mc_weights = []

    for i in range(len(algo_pos)):
        mc_weights.append(np.random.permutation(p))
    
    mc_weights = pd.DataFrame(mc_weights, index=algo_weights.index, columns=algo_weights.columns)
    
    # calculate returns
    mc_ret = (returns.iloc[60:] * mc_weights.shift()).sum(axis=1) / (abs(mc_weights.shift()).sum(axis=1)/2)
    mc_cumret = np.nancumprod(mc_ret+1)
    tot_ret = (mc_cumret[-1] - mc_cumret[0])/mc_cumret[0]
    total_returns.append(tot_ret)


# In[ ]:


total_returns = np.array(total_returns)
(total_returns>0.64).sum()/10000


# In[ ]:





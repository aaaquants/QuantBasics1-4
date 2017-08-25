import pandas as pd
import pandas_datareader.data as web
from dateutil.parser import parse
import cPickle
import pylab as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prices(tickers,start,end,backend='google'):
    if backend == 'quantopian':
        p = get_pricing(tickers,start,end)
        field = 'price'

    elif backend == 'google':
        p = web.DataReader(tickers, 'google', parse(start), parse(end)).ffill()
        field = 'Close'
        cPickle.dump(p,open('prices.pick','w'))

    elif backend == 'random':
        field = 'Close'
        p = web.DataReader(tickers, 'google', parse(start), parse(end)).ffill()
        for ticker in tickers:
            p[field][ticker] = np.cumsum(np.random.randn(len(p[field][ticker]))-0.0)+500

    elif backend == 'file':
        p = cPickle.load(open('prices.pick'))
        field = 'Close'


    # pp=pd.DataFrame(p[field],index=p[field].index,columns = tickers)
    scaled = MinMaxScaler((0,1)).fit_transform(p[field])
    pp=pd.DataFrame(scaled,index=p[field].index,columns = tickers)
    return pp

def calc_signals(tickers,p,a,b):
    sma = p.rolling(a).mean()
    smb = p.rolling(b).mean()
    signal = np.sign(sma - smb).diff()
    actual_signals = signal.dropna(how='all',axis=0)
    for col in actual_signals.columns:
        idx = actual_signals[col].first_valid_index()
        signal[col][idx] = signal[col][idx]/2.
    return signal

def calc_pnl(sig,p):
    sig_up = sig.cumsum().apply(lambda x:x*(x>0))
    sig_dwn = sig.cumsum().apply(lambda x:x*(x<0))
    pnlx = np.cumsum(p.diff()*sig_up+p.diff()*sig_dwn).sum(axis=1)
    return pnlx

def calc_sharpe(pnl):
    retsx = diff(pnl)
    retsx = retsx[~np.isinf(retsx)]
    sharpe = np.nanmean(retsx)/np.nanstd(retsx)*np.sqrt(252)
    return sharpe

def plot_sharpe():
    k = np.random.randn(1000)+0.08
    p = np.cumsum(k)+100
    p_sorted = np.cumsum(np.sort(k))+100
    print np.mean(k)/np.std(k)*np.sqrt(252)

    plt.plot(p)
    plt.plot(p_sorted)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.show()

def test_pnl():
    rets = np.zeros(1000)
    rets[500] = 1
    pr = np.cumsum(rets)

    df = pd.DataFrame(pr)
    sig = calc_signals(1,df,10,20)
    pnl = calc_pnl(sig,df)
    plt.subplot(3,1,1)
    plt.plot(df)
    plt.ylabel('price')
    plt.subplot(3,1,2)
    plt.plot(sig)
    plt.ylabel('signal')
    plt.subplot(3,1,3)
    plt.plot(pnl)
    plt.ylabel('pnl')
    plt.xlabel('time')
    plt.show()


if __name__=="__main__":
    BACKEND = 'google'
    # BACKEND = 'file'
    tickers = ['AAPL','MSFT','CSCO','XOM']
    start = '2003-01-01'
    end = '2017-06-01'
    p = prices(tickers,start,end,backend=BACKEND)
    test_pnl()
    plot_sharpe()

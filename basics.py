import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from dateutil.parser import parse
import cPickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabaz_score
import datetime
from ipdb import set_trace
import scipy

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
        p = p.loc[:,parse(start):parse(end)]
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
    retsx = np.diff(pnl)
    retsx = retsx[~np.isinf(retsx)]
    sharpe = np.nanmean(retsx)/np.nanstd(retsx)*np.sqrt(252)
    return sharpe

def calc_ddwn(pnl):
    hwm = 0
    ddwn = []
    for i in pnl:
        if i>hwm:
            hwm = i
        ddwn.append(i - hwm)
    return np.nanmin(ddwn)

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

def test_ddwn():
    k = np.random.randn(1000)
    ddwn = calc_ddwn(np.cumsum(k))
    plt.plot(np.cumsum(k))
    plt.xlabel('time')
    plt.ylabel('price')
    plt.title('drawdown: %s'%ddwn)
    plt.show()

def run_single(tickers, p):
    sig = calc_signals(tickers,p,10,20)
    pnl = calc_pnl(sig,p)
    sharpe = calc_sharpe(pnl)
    ddwn = calc_ddwn(pnl)
    return pnl,sharpe,ddwn

def parameter_sweep(tickers,p,params,N,progress = False):
    pnls = []
    sharpes = []
    ddwns = []
    new_params = []
    for i in range(N):
        if progress: print i
        a = min(params[i])
        b = max(params[i])
        if a==b: continue
        try:
            sig = calc_signals(tickers,p,a,b)
            pnl = calc_pnl(sig,p)
            pnls.append(pnl[-1])
            sharpes.append(calc_sharpe(pnl))
            ddwns.append(calc_ddwn(pnl))
            new_params.append([a,b])

        except:
            pnls.append(np.nan)
            sharpes.append(np.nan)
            ddwns.append(np.nan)
            new_params.append([a,b])

    return pnls,sharpes,ddwns,new_params

def run_parameter_sweep(tickers,start,end,BACKEND,N):
    sm = 5
    lm = 250
    frac = 0.7
    mid_point = str(datetime.timedelta((parse(end)-parse(start)).days*frac)+parse(start))
    print 'MID POINT:', mid_point
    print 'BACKEND:',BACKEND
    params = np.array([np.random.randint(sm,lm,(N,)) for i in range(2)]).T
    # plt.plot(params[:,0],params[:,1],'o')
    # plt.xlabel('parameter 1')
    # plt.ylabel('parameter 2')
    # plt.show()

    p0 = prices(tickers,start,mid_point,backend=BACKEND)
    pnls1,sharpes1,ddwns1,new_params = parameter_sweep(tickers,p0,params,N,progress=True)

    p1 = prices(tickers,mid_point,end,backend=BACKEND)
    pnls2,sharpes2,ddwns2,new_params = parameter_sweep(tickers,p1,params,N,progress=True)
    return pnls1,sharpes1,ddwns1,pnls2,sharpes2,ddwns2,new_params

def plot_pnl_hist(pnls1,pnls2):
    plt.hist(pnls1,40)
    plt.hist(pnls2,40)
    mean1 = np.mean(pnls1)
    mean2 = np.mean(pnls2)
    plt.xlabel('pnl')
    plt.ylabel('N')
    plt.title('mean train: %s; mean test: %s'%(mean1,mean2))
    plt.show()

def show_train_test_correlation(pnls1,pnls2):
    plt.plot(pnls1,pnls2,'o')
    plt.xlabel('train pnl')
    plt.ylabel('test pnl')
    plt.title('train-test correlation')
    plt.show()

def get_cluster_number(X):
    score = 0
    best_cluster_number = 0
    for i in range(2,10):
        # kmeans = AgglomerativeClustering(n_clusters = i).fit(X)
        kmeans = KMeans(n_clusters=i).fit(X)
        chs = calinski_harabaz_score(X,kmeans.labels_)
        print 'cluster number -->', i, chs
        if chs>score:
            best_cluster_number = i
            score = chs
    return best_cluster_number-1

def plot_clusters(pnls1,pnls2):
    Nc = get_cluster_number(np.array([pnls1,pnls2]).T)
    kmeans = KMeans(n_clusters=Nc+1).fit(np.array([pnls1,pnls2]).T)
    cPickle.dump(kmeans,open('kmeans.pick','w'))
    # plt.scatter(pnls1,pnls2,c=kmeans.labels_);
    # cPickle.dump([pnls1,pnls2],open('pnls.pick','w'))
    # plt.xlabel('train pnl')
    # plt.ylabel('test pnl')
    # plt.title('train-test correlation')
    # plt.show()
    return kmeans,Nc

def plot_linreg(x,y):
    m = np.polyfit(x,y,1)
    xx = np.linspace(min(x),max(x),1000)
    yy = np.polyval(m,xx)
    plt.plot(xx,yy)
    return m

def find_best_cluster(kmeans):
    median_oos_pnl = []
    for label in np.unique(kmeans.labels_):
        median_pnl = np.median(np.array(pnls2)[kmeans.labels_==label])
        median_oos_pnl.append(median_pnl)
    center_mean = np.argmax(np.mean(kmeans.cluster_centers_,axis=1))
    opt_label = np.argmax(median_oos_pnl)
    if center_mean!=opt_label:
        print 'Warning: best center mean is different from median oos pnl'
    return opt_label

def plot_best_cluster(kmeans):
    opt_label = find_best_cluster(kmeans)
    x = np.array(pnls1)[kmeans.labels_==opt_label]
    y = np.array(pnls2)[kmeans.labels_==opt_label]
    plt.scatter(x,y)
    m = plot_linreg(x,y)
    plt.title('slope of regression: %s'%m[0])
    plt.xlabel('train pnl')
    plt.ylabel('test pnl')
    return opt_label

def plot_best_parameters(kmeans):
    opt_label = find_best_cluster(kmeans)
    params = cPickle.load(open('params.pick'))
    plt.plot(np.array(params)[:,0],np.array(params)[:,1],'bo')
    plt.plot(np.array(params)[kmeans.labels_==opt_label,0],np.array(params)[kmeans.labels_==opt_label,1],'ro')
    plt.xlabel('parameter 1')
    plt.ylabel('parameter 2')

def get_best_parameters(params,pnls,N):
    idx = np.argsort(pnls)
    unique_params = np.unique(np.array(params)[idx][-N:],axis=0)
    return unique_params

def plot_best_params(params,tickers,start,end,BACKEND):
    p = prices(tickers,start,end,backend=BACKEND)
    for par in params:
        print par
	pnl = (calc_pnl(calc_signals(tickers,p,min(par),max(par)),p))
	plt.plot(p.index,(calc_pnl(calc_signals(tickers,p,min(par),max(par)),p)))
    plt.xlabel('time')
    plt.ylabel('PnL')

def plot_response_surface(pnls,params,tickers,start,end,backend='file'):
    from mpl_toolkits.mplot3d import Axes3D
    p = prices(tickers,start,end,backend=backend)
    best_params = get_best_parameters(params,pnls,50)
    x = []
    y = []
    z = []
    for par in best_params:
        x.append(par[0])
        y.append(par[1])
	z.append(calc_pnl(calc_signals(tickers,p,min(par),max(par)),p)[-1])
    n_points = 50
    X,Y = np.meshgrid(np.linspace(min(x),max(x),n_points),np.linspace(min(y),max(y),n_points))
    Z = scipy.interpolate.griddata(np.array([x,y]).T,np.array(z),(X,Y),method='cubic')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x,y,z,'ro')
    ax.plot_wireframe(X,Y,Z)
    plt.xlabel('x');plt.ylabel('y')

if __name__=="__main__":
    # BACKEND = 'google'
    BACKEND = 'file'
    tickers = ['AAPL','MSFT','CSCO','XOM']
    start = '2003-01-01'
    end = '2017-06-01'
    # p = prices(tickers,start,end,backend=BACKEND)
    # pnl,sharpe,ddwn = run_single(tickers,p)
    # plt.plot(pnl)
    # test_pnl()
    # test_ddwn()
    # plot_sharpe()
    # pnls1,sharpes1,ddwns1,pnls2,sharpes2,ddwns2,params = run_parameter_sweep(tickers,start,end,BACKEND,30000)
    # cPickle.dump([pnls1,pnls2],open('pnls.pick','w'))
    # cPickle.dump(params,open('params.pick','w'))
    # cPickle.dump([sharpes1,sharpes2],open('sharpes.pick','w'))
    # cPickle.dump([ddwns1,ddwns2],open('ddwns.pick','w'))
    # plot_pnl_hist(pnls1,pnls2)
    # plot_pnl_hist(pnls1,pnls2)
    # show_train_test_correlation(pnls1,pnls2)
    pnls1,pnls2 = cPickle.load(open('pnls.pick'))
    params = cPickle.load(open('params.pick'))
    # kmeans,Nc = plot_clusters(pnls1,pnls2)
    # opt_label = plot_best_cluster(kmeans)
    # plot_best_parameters(kmeans)
    # plt.show()
    # best_params = get_best_parameters(params,pnls1,15)
    # plot_best_params(best_params,tickers,start,end,BACKEND)
    plot_response_surface(pnls1,params,tickers,start,end)
    # importlib.import_module('mpl_toolkits').__path__
    plt.show()




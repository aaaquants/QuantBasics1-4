import cPickle
import scipy
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pnls = cPickle.load(open('pnls.pick'))
sharpes = cPickle.load(open('sharpes.pick'))
ddwns = cPickle.load(open('ddwns.pick'))

def get_score():

    assert(len(ddwns[0])==len(sharpes[0])==len(pnls[0]))

    score = []
    i = 0
    for pnl,shrp,ddwn in zip(pnls[0],sharpes[0],ddwns[0]):
        if i > 10: break
        if (pnl>1 and shrp>1) and ddwn>-2:
            print 'pnl:',round(pnl,2), '| ddwn:',round(ddwn,2)
            score.append(raw_input('what is your score:'))
            i += 1
        else:
            score.append(0)
        cPickle.dump(score,open('score.pick','w'))

def calc_scores():
    score = cPickle.load(open('score.pick'))
    # model = SVR()
    model = DecisionTreeRegressor()
    # model = LinearRegression()
    model.fit(np.array([pnls[0],ddwns[0]]).T[:len(score)],np.array(score))
    return model

def print_all_scores(model):
    x = []; y = []; z = []
    for pnl,shrp,ddwn in zip(pnls[0],sharpes[0],ddwns[0]):
        predicted_score = model.predict(np.array([[pnl,ddwn]]))
        if pnl > 3.5:

            # plt.plot(pnl,predicted_score,'bo')
            print pnl,shrp,ddwn,predicted_score
            x.append(pnl)
            y.append(ddwn)
            z.append(predicted_score[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.array(x),np.array(y),scipy.stats.rankdata(x)+scipy.stats.rankdata(y),'ro')
    # ax.plot(np.array(x),np.array(y),np.array(z),'bo')
    plt.xlabel('pnl')
    plt.ylabel('ddwn')

    plt.show()

# get_score()
model = calc_scores()
print_all_scores(model)

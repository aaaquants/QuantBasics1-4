import cPickle
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

pnls = cPickle.load(open('pnls.pick'))
sharpes = cPickle.load(open('sharpes.pick'))
ddwns = cPickle.load(open('ddwns.pick'))

def get_score():

    assert(len(ddwns[0])==len(sharpes[0])==len(pnls[0]))

    score = []
    for pnl,shrp,ddwn in zip(pnls[0],sharpes[0],ddwns[0]):
        if (pnl>1 and shrp>1) and ddwn>-2:
            print round(pnl,2),round(shrp,2),round(ddwn,2)
            score.append(raw_input('what is your score:'))
        else:
            score.append(0)
    cPickle.dump(score,open('score.pick','w'))

def calc_scores():
    score = cPickle.load(open('score.pick'))
    # model = SVR()
    model = DecisionTreeRegressor()
    model.fit(np.array([pnls[0],sharpes[0],ddwns[0]]).T,np.array(score))
    return model

def print_all_scores(model):
    for pnl,shrp,ddwn in zip(pnls[0],sharpes[0],ddwns[0]):
        predicted_score = score = model.predict(np.array([[pnl,shrp,ddwn]]))
        if predicted_score > 0.8: print pnl,shrp,ddwn,predicted_score

model = calc_scores()
print_all_scores(model)

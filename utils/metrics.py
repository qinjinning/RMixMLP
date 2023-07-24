import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))



def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def SCORE(pred, true):
    losses = []
    a = len(pred)
    for i in range(len(pred)):
        if pred[i]-true[i] < 0:
            losses.append(np.e**(-1*(pred[i]-true[i])/13)-1)
        else:
            losses.append(np.e**((pred[i]-true[i])/10)-1)
    loss = np.sum(losses)
    return loss


def metric(pred, true):
    score = SCORE(pred, true)
    rmse = RMSE(pred, true)
    mae = MAE(pred, true)

    return mae, score, rmse

def score_(x):
    losses = []
    for i in range(len(x)):
        if x[i] < 0:
            losses.append(np.e**(-1*x[i]/13)-1)
        else:
            losses.append(np.e**(x[i]/10)-1)
    return losses

def rmse_(x):
    return np.abs(x)

def metric2(x):

    return score_(x), rmse_(x)

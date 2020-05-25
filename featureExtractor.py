from TagNames import *
import numpy as np
from IPython import embed
# All 'data' arguments are assumed to be tuples of (start_time, end_time, type)

# return the average time that a signal does not have "NO_SIGNAL"
def averageOn(data):
    earliest = data[0][0]
    latest = data[-1][1]
    duration = latest - earliest

    if duration <= 0.:
        return 0.

    timeOn = 0
    for d in data:
        if (d[2] != NO_SIGNAL):
            timeOn += d[1] - d[0]

    avgOn = timeOn / duration
    return avgOn

def numberOfOccurances(data):
    num = 0
    for d in data:
        if (d[2] != NO_SIGNAL):
            num += 1

    return num / 1000

def variabilityOfSignal(data):
    earliest = data[0][0]
    latest = data[-1][1]
    duration = latest - earliest

    if (duration <= 0.):
        return 0.

    timesSwitched = 0
    value = data[0][2]
    for d in data:
        if value != d[2]:
            timesSwitched += 1
            value = d[2]

    return (timesSwitched / duration)

# For each datum, how many of the signals are active? 
def signalConcurrency(all_data):
    min_t = float("inf")
    max_t = float("-inf")
    for i, (k,v) in enumerate(all_data.items()):
        if (v[0][0] < min_t):
            min_t = v[0][0]
        elif (v[-1][1] > max_t):
            max_t = v[-1][1]


    n = 30
    intervals = np.linspace(min_t, max_t, n)
    t0 = intervals[0]
    idx = 1

    res = [] # a true or false for whether 2 or more signals are active at one time

    for i in range(len(intervals) - 1):
        t1 = intervals[i]

        votes = []
        for j, (k,v) in enumerate(all_data.items()):
            # TODO: this really should be more granular, but for now we just take the first valid entry in the interval range
            # is there an active signal in this time period?
            vote = 0
            for d in v:
                if (d[0] >= t0 and d[1] <= t1):
                    vote = (vote | 1) if d[1] is not NO_SIGNAL else (vote | 0)
                    # vote +=  1 if d[1] is not NO_SIGNAL else 0
                elif (d[1] > t1):
                    break
            # vote = round(vote / n, 3)
            votes.append(vote)

        if (sum(votes) >= 2):
            res.append(1)
        else:
            res.append(0)

        t0 = t1

    ret = round(np.mean(res), 3)
    # print(ret)
    return ret


# apply an feature extract to all streams of a user
def extractFeature(allData, feature_fncs):
    features = []

    for fnc in feature_fncs:
        if (fnc == signalConcurrency): # Special case for connecting the different streams
            features.append(fnc(allData))
        else:
            for i, (k,v) in enumerate(allData.items()):
                features.append(fnc(v))
        

    return features
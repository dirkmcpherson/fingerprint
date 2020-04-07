from TagNames import *
# All 'data' arguments are assumed to be tuples of (start_time, end_time, type)

# return the average time that a signal does not have "NO_SIGNAL"
def averageOn(data):
    earliest = data[0][0]
    latest = data[-1][1]
    duration = latest - earliest

    timeOn = 0
    for d in data:
        if (d[2] != NO_SIGNAL):
            timeOn += d[1] - d[0]

    return (timeOn / duration)

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

    timesSwitched = 0
    value = data[0][2]
    for d in data:
        if value != d[2]:
            timesSwitched += 1
            value = d[2]

    return (timesSwitched / duration)


# apply an feature extract to all streams of a user
def extractFeature(allData, feature_fncs):
    features = []
    for i, (k,v) in enumerate(allData.items()):
        [features.append(feature_fnc(v)) for feature_fnc in feature_fncs]

    return features
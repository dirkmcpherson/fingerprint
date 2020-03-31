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


# apply an feature extract to all streams of a user
def extractFeature(allData, feature_fnc):
    features = []
    for i, (k,v) in enumerate(allData.items()):
        features.append(feature_fnc(v))

    return features
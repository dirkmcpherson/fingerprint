import os 
import sys
import glob
import xml.etree.ElementTree as ET
import numpy as np
import random
from IPython import embed
from sklearn import svm

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

import featureExtractor as feature
# Tag name constants in separate file
from TagNames import *

DEBUG=True

def main():
    glob.glob("")

def parseForUser(fullPath):
    filename = fullPath.split("/")[-1]
    user = filename.split(".")[1] # Grab the user ID (a letter for the AMI dataset)

    return user


def dataForUser(uid, dataType, subtype=None):
    if (subtype is None):
        subtype = dataType

    ret = []
    fns = users[uid]
    targetFns = [f for f in fns if (dataType in f)]

    for fn in targetFns:
        root = ET.parse(fn).getroot()
        ret.append([entry for entry in root.findall(subtype)])
    return ret

def formatHeadDataForPlotting(head, placeHolderValue=1):
    # ranges = []
    timeSpansForSignalType = dict()
    skippedNoComms = 0
    for h in head:
        ts = h.get(START_TIME)
        tf = h.get(END_TIME)
        signalType = h.get(TYPE)
        if (ts is None) or (tf is None):
            continue
        elif (signalType is not None) and (signalType == NO_COMM_HEAD):
            skippedNoComms += 1
            continue

        timespan = (float(ts), float(tf))
        if (signalType in timeSpansForSignalType):
            timeSpansForSignalType[signalType].append(timespan)
        else:
            timeSpansForSignalType[signalType] = []

    # assumed to be in order
    return plotTimespansAsBinary(placeHolderValue)

def plotTimespansAsBinary(ranges, placeHolderValue=1.0):
    # assumed to be in order
    earliest = ranges[0][0]
    latest = ranges[-1][1]

    numSamples = 2 * len(ranges)
    inc = latest / numSamples
    xs = [i*inc for i in range(numSamples)]
    ys = []

    rangeIdx = 0
    targetRange = ranges[rangeIdx]
    for x in xs:
        y = -1
        # if the x doesn't fall inside a described range, its 0
        while (x > targetRange[1]):
            rangeIdx += 1
            targetRange = ranges[rangeIdx]

        if (x < targetRange[0]):
            y = 0
        else:
            y = placeHolderValue

        ys.append(y)
    return xs, ys

# Take the data out of xml format and store it as an ordered list of tuples (start_time, end_time, type). Where the entry for type looks in the relevent subfield for the interaction modality (e.g. 'form' for movement)
def reformatData(data, subType=None):
    ret = []
    excludedTypes = [NO_COMM_HEAD, OFF_CAMERA, SIT]
    for d in data:
        ts = d.get(START_TIME)
        tf = d.get(END_TIME)
        signalType = d.get(TYPE)
        if (ts is None) or (tf is None):
            continue
        elif (signalType is None) or (signalType in excludedTypes): # words have no entry for none, movement and head have specific types that indicate no info 
            signalType = NO_SIGNAL

        # If we have a subtype, classify using that instead of type
        if (signalType is not NO_SIGNAL) and (subType is not None):
            # Sometimes signals that should have subtypes just dont.
            subTypeSignal = d.get(subType)
            signalType = subTypeSignal if (subTypeSignal is not None) else signalType

            if (signalType is None):
                print("IT AINT RIGHT!! {}".format(subType))

        ret.append((float(ts), float(tf), signalType))

    return ret

# Format all the data for a user as ordered tuples and put it in a dictionary by type
def aggregateDataForUser(uid):
    words = dataForUser(uid, WORDS, 'w')[0]
    head = dataForUser(uid, HEAD)[0]
    movement = dataForUser(uid, MOVEMENT)[0]

    words = reformatData(words)
    head = reformatData(head, subType=FORM)
    movement = reformatData(movement)

    # Words neesd to be further post processed because it does not include any "no_signal" entries. 
    # Go through and fill in all temporal gaps with NO_SIGNAL. Change existing signals to SPEECH
    tmp = []
    minT = head[0][0]
    maxT = head[-1][1]
    for w in words:
        t0 = w[0]
        tf = w[1]

        if (minT < t0):
            tmp.append( (minT, t0, NO_SIGNAL) )
            minT = tf + 0.001 # epsilon
        else:
            tmp.append( (t0, tf, SPEECH))
            minT = tf + 0.001

    # add a no signal cap if needed
    if (tmp[-1][1] < maxT):
        tmp.append( (tmp[-1][1], maxT, NO_SIGNAL))

    words = tmp
    return {WORDS: words, MOVEMENT: movement, HEAD: head}

def dataFromRange(start_time, end_time, allDataAllUsers):
    ret = dict()
    for i, (key, allData) in enumerate(allDataAllUsers.items()):
        # go through and pluck out data between t0 and tf
        dataTypes = dict()
        for i, (dataTypeName, val) in enumerate(allData.items()):
            subset = []
            for entry in val:
                t0 = entry[0]
                tf = entry[1]
                if (t0 >= start_time and tf <= end_time):
                    subset.append(entry)
                # catch the edge cases where an interval cuts through a start or end time
                elif (t0 < start_time and tf > start_time):
                    subset.insert(0, (start_time, tf, entry[2]) ) 
                elif (t0 < end_time and tf > end_time):
                    subset.append( (t0, end_time, entry[2]) ) # fill in with an entry that goes between the time periods with the signaltype
            dataTypes[dataTypeName] = subset
        ret[key] = dataTypes

    return ret

# print out the data available for a user given its dictionary
def printDataDescription(uid, allData):
    for i, (key, val) in enumerate(allData.items()):
        types = dict()
        data = val
        t0 = data[0][0]
        tf = data[-1][1]
        for entry in data:
            if entry[2] in types:
                types[entry[2]] += 1
            else:
                types[entry[2]] = 1

        print("User {} has {} data from {:0.2f} to {:0.2f}. ".format(uid, key, t0, tf))
        for i, (key, val) in enumerate(types.items()):
            print("         {} has {} entries".format(key, val))

if __name__ == "__main__":
    users = dict()
    users['A'] = []
    users['B'] = []
    users['C'] = []
    users['D'] = []

    searchString = "ES2008a" # ES2008 and ES2009 a-c meetings all have relevent data
    if len(sys.argv) > 1:
        searchString = sys.argv[1]
        print("Using input meeting set.")
    else:
        print("Using default meeting set.")
        
    searchString = "./**/" + searchString + "*"
    print("Searching for "+ searchString)
    allInstances = glob.glob(searchString, recursive=True)
    for i in allInstances:
        uid = parseForUser(i)
        if (uid in users):
            users[uid].append(i)
        else:
            pass
            # users[uid] = [i]

    if (len(users.keys()) == 0):
        print("ERROR: No data read in.")
        embed()
    else:
        '''
        Signals we want:
            1) Talking duration
            2) Movement
            3) Head movement

        What we want from each signal
            1) duration of each "color"
            2) connections between signals (talking into head nod)
        '''
        plt.figure(0)
        idx = 1

        data = dict()
        for uid in users:
            data[uid] = aggregateDataForUser(uid)
            printDataDescription(uid, data[uid])
            # words = dataForUser(uid, WORDS, 'w')[0]
            # xs, ys = formatTimespanDataForPlotting(words, placeHolderValue=(idx*0.25))
            # head = dataForUser(uid, HEAD)[0]
            # xs, ys = formatHeadDataForPlotting(head, placeHolderValue=(idx*0.25))
            # plt.scatter(xs,ys)
            # idx += 1



        # Break up data into a bunch of snippets
        meetingDuration = data['A'][MOVEMENT][-1][1] # End time of last sample
        numSnippets = 8
        trainSize = 5
        testSize = numSnippets - 2
        results = np.zeros((4,4))
        
        numRuns = 100
        for i in range(numRuns):
            snippets = []
            t0 = 0
            tf = None
            for i in range(numSnippets):
                tf = (meetingDuration / numSnippets) * (i+1)
                snippets.append(dataFromRange(t0, tf, data))
                t0 = tf

            trainingSet = []
            testingSet = []

            for i in range(trainSize):
                idx = random.randint(0, len(snippets) - 1)
                selected = snippets[idx]
                trainingSet.append(selected)
                snippets.remove(selected)

            testingSet = snippets

            mapLetterToID = {'A':0, 'B':1, 'C':2, 'D':3}
            mapIDToLetter = {0:'A', 1:'B', 2:'C', 3:'D'}
            clf = svm.SVC(decision_function_shape='ovo')

            #SVM fit will forget everything it knew, so you have to average the features from all the sets for training and testing
            X = []
            Y = []
            for snippet in trainingSet:
                for i, (k,v) in enumerate(snippet.items()):
                    X.append(feature.extractFeature(v, feature.averageOn))
                    Y.append(mapLetterToID[k])

            clf.fit(X,Y)

            y_true = []
            y_pred = []
            for snippet in testingSet:
                for i, (k,v) in enumerate(snippet.items()):
                    x_pred = [feature.extractFeature(v, feature.averageOn)]
                    y_pred.append(clf.predict(x_pred))
                    y_true.append(k)

            # y_pred = [mapIDToLetter[entry[0]] for entry in y_pred]
            y_true = [mapLetterToID[entry] for entry in y_true]

            for truth, prediction in zip(y_true, y_pred):
                results[(truth,prediction)] += 1


        embed()
        print(results)
        # firstHalf = dataFromRange(0, 800., data)
        # secondHalf = dataFromRange(801, 1020, data)
        # mapLetterToID = {'A':0, 'B':1, 'C':2, 'D':3}
        # X = []
        # Y = []
        # for i, (k,v) in enumerate(firstHalf.items()):
        #     allData = v
        #     features = feature.extractFeature(allData, feature.averageOn)
        #     print("Features for {} : {}".format(k,features))
        #     X.append(features)
        #     Y.append(mapLetterToID[k])

        # clf = svm.SVC(decision_function_shape='ovo')
        # clf.fit(X,Y)

        # x_pred = []
        # y_pred = None
        # for i, (k,v) in enumerate(secondHalf.items()):
        #     allData = v
        #     features = feature.extractFeature(allData, feature.averageOn)
        #     x_pred.append(features)
        # y_pred = clf.predict(x_pred)

        # embed()


        # # Example of how to parse a file
        # fn = users["A"][0]
        # root = ET.parse(fn).getroot()

        # # print the starttime for all the words of user A in the file
        # for entry in root.findall("w"):
        #     print(entry.get(START_TIME))



        # plt.plot(xs,ys)
        # plt.show()
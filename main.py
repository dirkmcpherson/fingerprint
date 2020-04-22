import os 
import sys
import glob
import xml.etree.ElementTree as ET
import numpy as np
import random
import time
from IPython import embed
from sklearn import svm
import itertools

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

import featureExtractor as feature
# Tag name constants in separate file
from TagNames import *

DEBUG=True

class fancyExperiment():
    def __init__(self):
        self.users = None
        self.config = None

    def parseforuser(self, fullPath):
        filename = fullPath.split("/")[-1]
        user = filename.split(".")[1] # Grab the user ID (a letter for the AMI dataset)

        return user

    def dataForUser(self, uid, dataType, subtype=None):
        if (subtype is None):
            subtype = dataType

        ret = []
        fns = self.users[uid]
        targetFns = [f for f in fns if (dataType in f)]

        for fn in targetFns:
            root = ET.parse(fn).getroot()
            ret.append([entry for entry in root.findall(subtype)])
        return ret

    def plotTimespansAsBinary(self, userData, placeHolderValue=1.0):
        xs = []
        ys = []
        labels = []
        numItems = float(len(userData.keys()))
        numSamples = 2000
        mint, maxt = userData[WORDS][0][0], userData[WORDS][-1][1]
        x = np.arange(mint, maxt, 0.5)
        for i, (k,dataStream) in enumerate(userData.items()):
            if len(labels) < len(userData.keys()):
                labels.append(k)

            y = []
            y_val = (i+1) / numItems

            rangeIdx = 0
            targetRange = dataStream[rangeIdx]

            for t in x:
                tmp = -1
                while(t > targetRange[1]):
                    rangeIdx += 1
                    targetRange = dataStream[rangeIdx]

                if (targetRange[2] == NO_SIGNAL):
                    tmp = 0
                else:
                    tmp = y_val
                y.append(tmp)

            print("%s has %d entries" % (k, len([_ for _ in y if _ > 0])))
            xs.append(x)
            ys.append(y)

        return xs, ys, labels

    # Take the data out of xml format and store it as an ordered list of tuples (start_time, end_time, type). Where the entry for type looks in the relevent subfield for the interaction modality (e.g. 'form' for movement)
    def reformatData(self, allData, subType=None):
        ret = []
        excludedTypes = [NO_COMM_HEAD, OFF_CAMERA, SIT]

        for dataset in allData:
            for d in dataset:
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


        # No go through and reset the times so the set of meetings is one continuous data stream #TODO: Inefficient
        first = ret[0]
        entryIdx = 1
        for i in range(len(ret) - 2):
            second = ret[entryIdx]

            if (second[0] < first[1]):
                t = first[1] # add the end time of the previous range to the entry being updated
                ret[entryIdx] = (t+second[0], t+second[1], second[2])

            first = second
            entryIdx += 1


        return ret

    # Format all the data for a user as ordered tuples and put it in a dictionary by type
    def aggregateDataForUser(self, uid):
        words = self.dataForUser(uid, WORDS, 'w')
        head = self.dataForUser(uid, HEAD)
        movement = self.dataForUser(uid, MOVEMENT)

        words = self.reformatData(words)
        head = self.reformatData(head, subType=FORM)
        movement = self.reformatData(movement)

        # extend the list of meetings into one long list. Entries are in order time starts over at the start of each file.
        # embed()
        # return

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

        # Further split HEAD data
        if (False):
            head_default = []
            head_nod = []
            head_shake = []

            for entry in head:
                gestureType = entry[2]
                negativePlaceholder = (entry[0], entry[1], NO_SIGNAL)
                if (gestureType == NO_SIGNAL):
                    head_default.append(entry)
                    head_nod.append(entry)
                    head_shake.append(entry)
                elif (gestureType == NOD):
                    head_nod.append(entry)
                    head_shake.append(negativePlaceholder)
                    head_default.append(negativePlaceholder)
                elif (gestureType == SHAKE):
                    head_shake.append(entry)
                    head_default.append(negativePlaceholder)
                    head_nod.append(negativePlaceholder)
                else:
                    head_default.append(entry)
                    head_shake.append(negativePlaceholder)
                    head_nod.append(negativePlaceholder)
            return {WORDS: words, MOVEMENT: movement, HEAD: head_default, HEAD_NOD: head_nod, HEAD_SHAKE: head_shake}
        else:
            return {WORDS: words, MOVEMENT: movement, HEAD: head}
            # return {WORDS:words}
            # return {WORDS:words, MOVEMENT: movement}
            # return {WORDS:words}

    def dataFromRange(self, start_time, end_time, allDataAllUsers):
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
    def printDataDescription(self, uid, allData):
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

    def run_experiment(self, config):
        self.config = config
        searchStrings = self.config["file_names"] # and ES2009 a-c meetings all have relevent data
        # if len(sys.argv) > 1:
        #     searchStrings = [sys.argv[1]]
        #     print("Using input meeting set.")
        # else:
        #     print("Using default meeting set.")

        Fscores = []
        true_positives = []
        false_positives = []
        activity_scores = [] # activity in each snippet
        for searchString in searchStrings:
            self.users = dict()
            self.users['A'] = []
            self.users['B'] = []
            self.users['C'] = []
            self.users['D'] = []

            NUM_USERS = len(self.users.keys())
            mapLetterToID = {'A':0, 'B':1, 'C':2, 'D':3}
            mapIDToLetter = {0:'A', 1:'B', 2:'C', 3:'D'}
                
            searchString = "./**/" + searchString + "*"
            print("Searching for "+ searchString)
            allInstances = glob.glob(searchString, recursive=True)
            for i in allInstances:
                uid = self.parseforuser(i)
                if (uid in self.users):
                    self.users[uid].append(i)
                else:
                    pass
                    # self.users[uid] = [i]

            if (len(self.users.keys()) == 0):
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
                for uid in self.users:
                    data[uid] = self.aggregateDataForUser(uid)
                    # printDataDescription(uid, data[uid])

                # Plot the profiles of each users
                # embed()
                self.plot(data)

                
                # Break up data into a bunch of snippets
                meetingDuration = data['A'][WORDS][-1][1] # End time of last sample
                numSnippets = self.config['snippet_size']
                trainSize = self.config['test_ratio']
                testSize = numSnippets - trainSize
                results = np.zeros((NUM_USERS,NUM_USERS))
                
                numRuns = self.config["num_runs"]
                for i in range(numRuns):
                    snippets = []
                    t0 = 0
                    tf = None
                    for i in range(numSnippets):
                        tf = (meetingDuration / numSnippets) * (i+1)
                        snippets.append(self.dataFromRange(t0, tf, data))
                        t0 = tf

                    trainingSet = []
                    testingSet = []

                    for i in range(trainSize):
                        idx = random.randint(0, len(snippets) - 1)
                        selected = snippets[idx]
                        trainingSet.append(selected)
                        # print("Training set got snippet from %f to %f" % (selected['A'][WORDS][0][0], selected['A'][WORDS][-1][1]))
                        snippets.remove(selected)

                    testingSet = snippets

                    # Check activity on the snippets
                    for snp in snippets:
                        for i, (k,v) in enumerate(snp.items()):
                            for _, (k,v) in enumerate(v.items()):
                                activity_scores.append(feature.averageOn(v))


                    clf = svm.SVC(decision_function_shape='ovo')
                    # clf = svm.LinearSVC()
                    f_fncs = self.config['feature_functions'] # = [feature.variabilityOfSignal, feature.averageOn] # feature.numberOfOccurances # 

                    #SVM fit will forget everything it knew, so you have to average the features from all the sets for training and testing
                    X = []
                    Y = []
                    for snippet in trainingSet:
                        for i, (k,v) in enumerate(snippet.items()):
                            X.append(feature.extractFeature(v, f_fncs))
                            Y.append(mapLetterToID[k])

                    # embed()
                    AVERAGE_ALL = False # Average all runs so there's just one data point for each user
                    if (AVERAGE_ALL):
                        tmp_x = []
                        tmp_y = []
                        d = dict()
                        for x,y in zip(X,Y):
                            if y in d:
                                d[y] = map(sum, zip(d[y], x)) # elementwise sum the two lists
                            else:
                                d[y] = x # initialize to first set of features

                        numEntries = len(X)
                        for i, (k,v) in enumerate(d.items()):
                            tmp_x.append([entry / float(numEntries) for entry in v])
                            tmp_y.append(k)

                        X = tmp_x
                        Y = tmp_y

                    clf.fit(X,Y)

                    y_true = []
                    y_pred = []
                    for snippet in testingSet:
                        for i, (k,v) in enumerate(snippet.items()):
                            x_pred = [feature.extractFeature(v, f_fncs)]
                            y_pred.append(clf.predict(x_pred))
                            y_true.append(k)

                    # y_pred = [mapIDToLetter[entry[0]] for entry in y_pred]
                    y_true = [mapLetterToID[entry] for entry in y_true]

                    for truth, prediction in zip(y_true, y_pred):
                        results[(truth,prediction)] += 1

                    # print("------------------------------------------------------------------")

                #normalize
                total = np.sum(results)
                # results /= total
                np.set_printoptions(precision=2)
                # print(searchString)
                # print(results)

                true_positive = np.sum(np.diag(results) )
                false_positive = total - true_positive
                false_negatives = false_positive # double-check: a row is a single true-positive entry and 3 false-positive entries (for other rows' IDs). A column is a single true-positive and 3 false-negative entries for the true-positive ID.

                true_positives.append(true_positive)
                false_positives.append(false_positive)

                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negatives)

                f_score = 2 * (precision * recall) / (precision + recall)
                Fscores.append(f_score)
                # print("--------------------------------------------------------------------------------------")

        print(Fscores)
        print("Mean f-score ", np.mean(Fscores))
        return Fscores, true_positives, false_positives, activity_scores

    def plot(self, data):
        if self.config['plot']:
            fig, ax = plt.subplots(len(data.keys()))
            plotIdx = 0
            for uid in data.keys():
                xs, ys, labels = self.plotTimespansAsBinary(data[uid])
                numPlots = len(xs)
                for j in range(len(xs)):
                    ax[plotIdx].scatter(xs[j],ys[j])

                if (plotIdx == 0):
                    ax[plotIdx].legend(labels)

                plotIdx += 1
            fig.suptitle("Binary Data Streams for Each User.")
            plt.show()
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

if __name__ == "__main__":
    exp = fancyExperiment()


    snippet_sizes = [i for i in range(4, 16, 2)]
    f_scores = []
    activity = []

    xs = []
    ys = []

    data = []
    
    # numExperiments = 10
    # for i in range(numExperiments):
        # fs = []
        # for snpsz in snippet_sizes:
    feature_functions = [[feature.variabilityOfSignal], [feature.averageOn], [feature.variabilityOfSignal, feature.averageOn]]
    snpsz = 12


    idx = 0
    roc = []
    for fncs in feature_functions:
        config = dict()
        config['plot'] = False
        config['snippet_size'] = snpsz
        config['test_ratio'] = int(np.floor(0.75 * snpsz))
        config['feature_functions'] = fncs # [feature.variabilityOfSignal, feature.averageOn]
        config["num_runs"] = 50 # how many times k-fold training/testing is done
        config["file_names"] = ["ES2009", "ES2008", "IS1008", "IS1009", "IS1003", "IS1004", "IS1005", "IS1006"]

        f, tp, fp, act = exp.run_experiment(config)


        total = sum(fp) + sum(tp)
        roc.append((sum(fp)/total, sum(tp)/total))

        activity.extend(act)    
        f_scores.extend(f)
        # data.append(list(zip(snippet_sizes, fs)))
        # for x, ylist in zip(snippet_sizes, f_scores):
        #     [xs.append(x) for y in ylist]
        #     ys.extend(ylist)

    print("Final FScores: ", (f_scores))

    for pair in roc:
        plt.scatter(pair[0], pair[1])
    plt.legend(["signalVar","avgOn", "avgOn+signalVar"])
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.show()

    plt.hist(activity, bins = [.01*i for i in range(101)])
    plt.title("Histogram of signal percentage-on in each snippet")
    plt.xlabel("Signal on %")
    plt.ylabel("# of instance")
    plt.show()

    # data has numExperiments zips of (snippet_size, F_score_on_each_file)
    # color each file respectively 
    # import matplotlib
    # numColors = len(config["file_names"])
    # colors_hsv = [( i * (1./numColors) ,1.,0.8) for i in range(numColors)]
    # colors_rgb = [matplotlib.colors.hsv_to_rgb(c) for c in colors_hsv]

    # fig = plt.figure()

    # for i in range(numColors):
    #     x = []
    #     y = []
    #     for zp in data:
    #         for snpsz, fscores in zp:
    #             x.append(snpsz)
    #             y.append(fscores[i])
    #     print(x,y)
    #     plt.scatter(x,y,c=[colors_rgb[i]])

    # plt.title("fscores for each meeting")
    # plt.xlabel("Snippets per meeting")
    # plt.ylabel("Fscore")
    plt.show()

    
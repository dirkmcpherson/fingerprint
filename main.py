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
from config import *

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
        time_offset = 0.
        tmp = []
        for entry in ret:
            second = entry
            if (second[0] < first[1]):
                time_offset += first[1]

            tmp.append((entry[0]+time_offset, entry[1]+time_offset, entry[2]))
            first = second

        # entryIdx = 1
        # for i in range(len(ret) - 2):
        #     second = ret[entryIdx]

        #     if (second[0] < first[1]):
        #         embed()
        #         time_offset += first[1] # Just make them sequential
        #         t = first[1] # add the end time of the previous range to the entry being updated
        #         ret[entryIdx] = (t+second[0], t+second[1], second[2])

        #     first = second
        #     entryIdx += 1

        ret = tmp
        # embed()
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

        # Words needs to be further post processed because it does not include any "no_signal" entries. 
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

    # Extract all data occurring between a start time and an end time.
    def dataFromRange(self, start_time, end_time, allDataAllUsers):
        ret = dict()
        for i, (uid, allData) in enumerate(allDataAllUsers.items()):
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
                # If there's no data, make an entry indicating so
                if (len(subset) == 0):
                    # print("No data for {} {} from {} to {}".format(key, dataTypeName, start_time, end_time))
                    subset.append((start_time, end_time, NO_SIGNAL))
                dataTypes[dataTypeName] = subset
            ret[uid] = dataTypes

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
        filenames = self.config["file_names"] # and ES2009 a-c meetings all have relevent data

        Fscores = []
        activity_scores = [] # activity in each snippet
        tp_fp_tn_fn = []
        for fn in filenames:
            self.users = dict()
            self.users['A'] = []
            self.users['B'] = []
            self.users['C'] = []
            self.users['D'] = []

            NUM_USERS = len(self.users.keys())
            mapLetterToID = {'A':0, 'B':1, 'C':2, 'D':3}
            mapIDToLetter = {0:'A', 1:'B', 2:'C', 3:'D'}
                
            fn = "./**/" + fn + "*"
            print("Searching for "+ fn)
            allInstances = glob.glob(fn, recursive=True)
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
                # self.plot(data)

                
                # Break up data into a bunch of snippets
                meetingDuration = np.max([data[UID][WORDS][-1][1] for UID in ['A','B','C','D']]) # End time of last time someone speaks TODO: improve
                snippetDuration = self.config['snippet_duration']
                test_ratio = self.config['test_ratio']

                print("DURATION: ", meetingDuration)
                shortestMeeting = 5279 # hardcoded fact
                leastSnippets = int(np.floor(shortestMeeting/snippetDuration))

                results = np.zeros((NUM_USERS,NUM_USERS))
                
                numRuns = self.config["num_runs"]
                for i in range(numRuns):
                    snippets = []
                    t0 = 0
                    tf = snippetDuration
                    while tf <= meetingDuration:
                        snippets.append(self.dataFromRange(t0, tf, data))
                        t0 = tf
                        tf += snippetDuration

                    testSize = int(np.floor(test_ratio * leastSnippets))
                    trainSize = leastSnippets - testSize
                    trainingSet = []
                    testingSet = []

                    for i in range(trainSize):
                        idx = random.randint(0, len(snippets) - 1)
                        selected = snippets[idx]
                        trainingSet.append(selected)
                        snippets.remove(selected)

                    for i in range(testSize):
                        idx = random.randint(0, len(snippets) - 1)
                        selected = snippets[idx]
                        testingSet.append(selected)
                        snippets.remove(selected)

                    # print("Num snippets in test:train {}:{}".format(len(testingSet), len(trainingSet)))

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
                    # AVERAGE_ALL = False # Average all runs so there's just one data point for each user
                    # if (AVERAGE_ALL):
                    #     tmp_x = []
                    #     tmp_y = []
                    #     d = dict()
                    #     for x,y in zip(X,Y):
                    #         if y in d:
                    #             d[y] = map(sum, zip(d[y], x)) # elementwise sum the two lists
                    #         else:
                    #             d[y] = x # initialize to first set of features

                    #     numEntries = len(X)
                    #     for i, (k,v) in enumerate(d.items()):
                    #         tmp_x.append([entry / float(numEntries) for entry in v])
                    #         tmp_y.append(k)

                    #     X = tmp_x
                    #     Y = tmp_y

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

                #---- Compute precision and recall ----#

                # Each class has true positives in its diagnal slot, false negatives in its row, and false positives in its column
                # true_positive = np.sum(np.diag(results))

                for trueID in [0,1,2,3]:
                    tp = 0
                    fp = 0
                    tn = 0
                    fn = 0
                    for row in range(len(results)):
                        for col in range(len(results[0])):
                            if row == trueID and col == trueID:
                                tp = results[(row, col)]
                            elif row == trueID and col != trueID:
                                fn += results[(row, col)]
                            elif row != trueID and col == trueID:
                                fp += results[(row, col)]
                            else:
                                tn += results[(row, col)]


                    tp_fp_tn_fn.append((tp, fp, tn, fn))

                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)

                    # embed()

                    f_score = 2 * (precision * recall) / (precision + recall)
                    if (np.isnan(f_score)):
                        print("WARN: nan f-scored defaulting to 0")
                        f_score = 0
                        # embed()
                    else:
                        Fscores.append(f_score)
                #--------------------------------------#

        print(Fscores)
        print("Mean f-score ", np.mean(Fscores))
        return Fscores, tp_fp_tn_fn, activity_scores

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


    snippet_durations = [i for i in range(1, 100, 5)] # [60] #[i for i in range(5, 60, 5)]
    print(snippet_durations)
    f_scores = []
    accuracies = []
    activity = []

    xs = []
    ys = []

    data = []
    
    # numExperiments = 10
    # for i in range(numExperiments):
        # fs = []
        # for snpsz in snippet_sizes:
    feature_functions = [[feature.variabilityOfSignal, feature.averageOn], [feature.signalConcurrency, feature.averageOn], [feature.variabilityOfSignal, feature.signalConcurrency]]

    # autoconstruct strings:
    feature_function_strings = []
    for fncs in feature_functions:
        label = ""
        for f in fncs:
            l = None
            if (f == feature.variabilityOfSignal): 
                l = "signalVar"
            elif (f == feature.averageOn):
                l = "avgOn"
            elif (f == feature.signalConcurrency):
                l = "conc"

            if (len(label) == 0):
                label += l
            else:
                label += ("+" + l)
        feature_function_strings.append(label)
        
        

    # feature_function_strings = ["signalVar","avgOn", "avgOn+signalVar"]
    # feature_functions = [[feature.signalConcurrency]]
    # feature_function_strings = ["concurrency"]


    idx = 0
    fs = []
    tps_fps_tns_fns = []

    roc = []
    for fncs in feature_functions:
        curve = []
        acc = []
        # fncs = [feature.variabilityOfSignal, feature.averageOn]
        for duration in snippet_durations:
            config = dict()
            config['plot'] = False
            config['snippet_duration'] = duration
            config['test_ratio'] = 0.25
            config['feature_functions'] = fncs #[feature.variabilityOfSignal, feature.averageOn] # fncs # 
            config["num_runs"] = 20 # how many times k-fold training/testing is done
            config["file_names"] = ["ES2009", "ES2008", "IS1008", "IS1009", "IS1003", "IS1004", "IS1005", "IS1006"]

            f, tp_fp_tn_fn, act = exp.run_experiment(config) # tp_fp_tn_fn has one entry for each user (4) for each file (len(config["file_names"])). 

            acc_sum = []
            for u in tp_fp_tn_fn:
                tp_tn_avg = (u[0] + u[2]) / sum(u)
                acc_sum.append(tp_tn_avg)

            acc.append(sum(acc_sum)/len(tp_fp_tn_fn))

            # total = sum(fp) + sum(tp)
            # curve.append((duration, sum(fp)/total, sum(tp)/total))
            # embed()
            fp_vs_tp = [(entry[1] / sum(entry), entry[0] / sum(entry)) for entry in tp_fp_tn_fn]
            # print(fp_vs_tp)
            curve.append(fp_vs_tp)

            activity.extend(act)    
            f_scores.extend(f)
            fs.append(f)
            tps_fps_tns_fns.append(tp_fp_tn_fn)
        roc.append(curve)
        accuracies.append(acc)

    print("Final FScores: ", (f_scores))

# ------------------------------------- #
    # Plot accuracies by feature set and snippet duration
    for trial in accuracies:
        # each trial is a different set of features. each point in a trial is for a different snippet duration.
        plt.plot(snippet_durations, trial)

    plt.title("Classification accuracy by feature set and snippet duration. ")
    plt.ylabel("Accuracy")
    plt.xlabel("Snippet Durations")
    # plt.legend(["std", "avgOn+signalVar"])
    plt.legend(feature_function_strings)
    plt.show()

# ------------------------------------- #
    # Bar plot of f-scores for best snippet length
    y = [np.mean(feature_set) for feature_set in fs]
    x = [i for i in range(len(y))]
    plt.xticks(x, tuple(feature_function_strings))
    plt.title("F-score by feature set for {} second snippets.".format(snippet_durations[0]))
    plt.ylabel("F-Score")
    plt.bar(x,y)
    plt.show()

# ------------------------------------- #
    # reformat roc curve data
    # embed()
    allvals = []
    for variant in roc:
        variant.sort(key=lambda x: x[0])
        tp = []
        fp = []
        for snippetDuration in variant:
            fp.append(np.mean([entry[0] for entry in snippetDuration]))
            tp.append(np.mean([entry[1] for entry in snippetDuration]))
            # fp.extend([entry[0] for entry in snippetDuration])
            # tp.extend([entry[1] for entry in snippetDuration])

        for i in range(len(tp)):
            print("{:1.2f}, {:1.2f}".format(fp[i], tp[i]))
        
        allvals.extend(tp)
        allvals.extend(fp)
        plt.scatter(fp,tp)

    # plt.scatter([entry[0] for entry in roc], [entry[1] for entry in roc])
    highestVal = max(allvals)
    plt.plot(np.linspace(0,highestVal), np.linspace(0,highestVal), '--')

    plt.ylabel("True Positives")
    plt.xlabel("False Positives")
    # plt.legend(["std", "avgOn+signalVar"])
    plt.legend(["signalVar","avgOn", "avgOn+signalVar", "std"])

# -------------------------------------#
    # for pair in roc:
    #     plt.scatter(pair[0], pair[1])
    # plt.legend(["signalVar","avgOn", "avgOn+signalVar"])
    # plt.xlabel("False Positives")
    # plt.ylabel("True Positives")
    # plt.show()

# -------------------------------------#
    # plt.hist(activity, bins = [.01*i for i in range(101)])
    # plt.title("Histogram of signal percentage-on in each snippet")
    # plt.xlabel("Signal on %")
    # plt.ylabel("# of instance")
    # plt.show()

# -------------------------------------#
    # color each file respectively 
    # import matplotlib
    # numColors = len(config["file_names"])
    # colors_hsv = [( i * (1./numColors) ,1.,0.8) for i in range(numColors)]
    # colors_rgb = [matplotlib.colors.hsv_to_rgb(c) for c in colors_hsv]

    # # fig = plt.figure()

    # for i in range(numColors):
    #     x = []
    #     y = []
    #     for f_score in fs:
    #         x.append(snippet_durations[i])
    #         y.append(f_score[i])
    #     print(x,y)
    #     plt.scatter(x,y,c=[colors_rgb[i]])

    # plt.title("fscores for each meeting")
    # plt.xlabel("Snippets per meeting")
    # plt.ylabel("Fscore")
    plt.show()
    # embed()
    
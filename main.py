import os 
import sys
import glob
import xml.etree.ElementTree as ET
from IPython import embed

# Tag name constants in separate file
from TagNames import *

def main():
    glob.glob("")

def parseForUser(fullPath):
    filename = fullPath.split("/")[-1]
    user = filename.split(".")[1] # Grab the user ID (a letter for the AMI dataset)

    return user

def headGestures(users):
    for uid in users.keys():
        for fn in users[uid]:
            if "head" in fn:
                root = ET.parse(fn).getroot()
                for entry in root.findall(HEAD):
                    form = entry.get(FORM)
                    print("{}: {} - {}".format(uid, entry.get(TYPE), entry.get(FORM)))

        

if __name__ == "__main__":
    users = dict()

    if len(sys.argv) > 1:
        searchString = sys.argv[1]
        searchString = "./**/" + searchString + "*"
        print("Searching for "+ searchString)
        allInstances = glob.glob(searchString, recursive=True)

        for i in allInstances:
            uid = parseForUser(i)

            if (uid in users):
                users[uid].append(i)
            else:
                users[uid] = [i]



    if (len(users.keys()) == 0):
        print("ERROR: No data read in.")
    else:
        headGestures(users)
        # # Example of how to parse a file
        # fn = users["A"][0]
        # root = ET.parse(fn).getroot()

        # # print the starttime for all the words of user A in the file
        # for entry in root.findall("w"):
        #     print(entry.get(START_TIME))




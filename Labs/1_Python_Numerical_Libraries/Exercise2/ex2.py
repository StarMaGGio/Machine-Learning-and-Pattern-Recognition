# -*- coding: utf-8 -*-

import sys

class BusRecord:
    def __init__(self, busId, lineId, x, y, t):
        self.busId = busId
        self.lineId = lineId
        self.x = x
        self.y = y
        self.t = t
        
def loadAllRecords(file):
    try:
        listRecords = []
        with open(file) as f:
            for line in f:
                busId, lineId, x, y, t = line.split()
                newRecord = BusRecord(busId, lineId, int(x), int(y), int(t))
                listRecords.append(newRecord)
        return listRecords
    except:
        raise # exception propagated
        
def euclideanDistance(r1, r2):
    return ((r1.x-r2.x)**2 + (r1.y-r2.y)**2)**0.5
        
def computeBusTotalDistanceTime(listRecords, busId):
    # first filter records to keep only passed busId and sort by the time parameter
    busIdRecords = sorted([b for b in listRecords if b.busId == busId], key = lambda v: v.t)
    
    if len(busIdRecords) == 0:
        return None
    
    totalDistance = 0.0
    totalTime = 0.0
    for r1, r2 in zip(busIdRecords[:-1], busIdRecords[1:]):
        totalDistance += euclideanDistance(r1, r2)
    totalTime = busIdRecords[-1].t - busIdRecords[0].t
    return totalDistance, totalTime

def computeAvgSpeed(listRecords, lineId):
    filteredRecords = [r for r in listRecords if r.lineId == lineId]
    busIdSet = set([r.busId for r in filteredRecords]) # return only univoque busIds
    
    if len(busIdSet) == 0:
        return 0.0
    
    totDist = 0.0
    totTime = 0.0
    for busId in busIdSet:
        d, t = computeBusTotalDistanceTime(filteredRecords, busId)
        totDist += d
        totTime += t
    return totDist/totTime
    
if __name__ == '__main__':
    listRecords = loadAllRecords(sys.argv[1])
    
    if sys.argv[2] == '-b':
        print('%s - Total Distance: ' % sys.argv[3], computeBusTotalDistanceTime(listRecords, sys.argv[3])[0])
    elif sys.argv[2] == '-l':
        print('%s - Avg speed: ' % sys.argv[3], computeAvgSpeed(listRecords, sys.argv[3]))
    else:
        raise KeyError()
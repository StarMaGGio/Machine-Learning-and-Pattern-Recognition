# -*- coding: utf-8 -*-

import sys

monthNames = {
    1: 'January',
    2:'February',
    3:'March',
    4:'April',
    5:'May',
    6:'June',
    7:'July',
    8:'August',
    9:'September',
    10:'October',
    11:'November',
    12:'December'
    }

if __name__ == '__main__':
    try:
        f = open(sys.argv[1], 'r')
    except:
        print("Error opening the file")
        sys.exit(1)
        
    birthsInCity = {} # city -> num of births
    birthsInMonth = {} # month -> num of births
        
    for line in f:
        name, surname, city, date = line.split()
        day, month, year = date.split("/")
        month_int = int(month)
        
        if city not in birthsInCity:
            birthsInCity[city] = 0
        birthsInCity[city] += 1
        
        if month_int not in birthsInMonth:
            birthsInMonth[month_int] = 0;
        birthsInMonth[month_int] += 1;
    
    f.close()
    
    print("Births per city:")
    for city in birthsInCity:
        print('\t%s: %d' % (city, birthsInCity[city]))
    print("Births per month: ")
    for month_int in birthsInMonth:
        print('\t%s: %d' % (monthNames[month_int], birthsInMonth[month_int]))
    print('Average number of births: %.2f' % (float(sum(birthsInCity.values()))/float(len(birthsInCity.keys()))))
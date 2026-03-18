import numpy

def loadFile(fileName):
    f = open(fileName, 'r')
    N = int(f.readline())
    lightCoords = []
    for line in f:
        x, y = line.split()
        lightCoords.append((int(x), int(y)))
    f.close()
    return N, lightCoords
    
if __name__ == '__main__':
    N, lightCoords = loadFile('ex5_data.txt')
    matrix = numpy.zeros((N, N))
    for x, y in lightCoords:
        matrix[x, y] += 0.5
        matrix[max(x-1,0):x+2, max(y-1,0):y+2] += 0.3
        matrix[max(x-2,0):x+3, max(y-2,0):y+3] += 0.2
    print(matrix)
import sys

def computeFinalScore(listScores):
    return sum(sorted(listScores)[1:-1])

# Competitor class that will contain data from the txt input datas
class Competitor:
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores
        self.final_score = computeFinalScore(self.scores)
        
if __name__ == "__main__":
    
    listBestCompetitors = []
    highestCountryScore = {}
    
    with open(sys.argv[1]) as f: # open the txt input file
        for line in f:
            # Read data from the input line
            name, surname, country = line.split()[0:3] # Split and take from 0 included to 3 excluded
            scores = line.split()[3:] # Split and take all the elements from 3 included to the end of the list
            scores = [float(i) for i in scores] # Fast method to convert all the elements into float values
            
            comp = Competitor(name, surname, country, scores) # Build the current competitor
            listBestCompetitors.append(comp) # Append the new competitor to the list
            
            if len(listBestCompetitors) >= 4:
                # Sort the list on the final_score of the competitors, reverse the list, select only the first 3 values
                listBestCompetitors = sorted(listBestCompetitors, key = lambda c: c.final_score)[::-1][0:3]
                
            if comp.country not in highestCountryScore:
                highestCountryScore[comp.country] = 0 # Add the new country to the dictionary
            highestCountryScore[comp.country] += comp.final_score # Sum the current competitor final score to the total of his country
            
    # Check if there were competitors in the list
    if len(highestCountryScore) == 0:
        print('No competitors')
        sys.exit(0)
        
    bestCountry = None
    # Iterare over all the countries and select the one with the best total score
    for c in highestCountryScore:
        if bestCountry is None or highestCountryScore[c] > highestCountryScore[bestCountry]:
            bestCountry = c
            
    # Output results
    print('final ranking:')
    for pos, comp in enumerate(listBestCompetitors):
        print('%d: %s %s - Score: %.1f' % (pos, comp.name, comp.surname, comp.final_score))
    print()
    print('Best Contry: ')
    print('%s - Total score: %.1f' % (bestCountry, highestCountryScore[bestCountry]))
import pandas
import math
import re

from pyprose.matching.text import learn_patterns

DATABASEPATH = "DGov_Typo/BLM_UT_National_Scenic_and_Historic_Trails_(Arc)/"
SIGNIFICANCETHRESHHOLD = 0.1





def main():
   db_in =pandas.read_csv("dirty.csv")
   db = db_in.astype(str)
   examples = [
      "Ind-674-PRO",
      "US-823-JUN",
      "US-238-JUN",
      "QUAL-47",
      "QUAL-21",
      "Zim-843-PRO",
      "Eng-781-JUN",
      "Aus-664-PRO",
      "QUAL-88",
      "Ind-473-JUN",
      "Eng-573-JUN",
      "Zim-392-PRO",
      "QUAL-10",
      "A2.A3.A4.",
      "A2.A3.",
      "AAA3",
      "A5.A7.",
      "A8.A9.",
      "A3.A4.A5.",
      "A7.",
      "A6.A2.A3.",
      "A4.A3.A2.",
      "A9.A2.A3.",
      "A2.A3.A4."
      "A2.A4.A6."
   ]
   patterns = ["^QUAL-[0-9]{2}$", "^[A-Z][a-z]+-[0-9]{3}-[A-Z]{3}$", "^A[0-9]\.A[0-9]\.A[0-9]\.$", "^A[0-9]\.A[0-9]\.$"]
   #db = db.iloc[:,[4]]
   #DataVinci(db)
   #patterns = getPatterns(examples)
   print(patterns)
   pattern = str(patterns[1])
   Dags = generateDAG(pattern[::-1],"usa_837")
   print(Dags)
   Dag = ['A','[0-9]','\.','A','[0-9]','\.']
   entry = "AAA3"
   moves,costs = generateMatrices(Dag,entry)
   fillMatrices(Dag,entry,moves,costs)
   printMatrix(moves,len(entry) + 1, len(Dag) + 1)
   printMatrix(costs,len(entry) + 1, len(Dag) + 1)
   
   candidate = readMovesMatrix(moves,entry,Dag,len(entry), len(Dag))
   print(candidate)



def DataVinci(data: pandas.DataFrame):
   for columnName in data.columns:
      column = data[columnName]
      patterns = getPatterns(column)
      print(patterns)
      llmPatterns = getLLMPatterns(column,patterns)
      columnDF = column.to_frame()
      for index,row in columnDF.iterrows():
         dirty = True
         entry = row[columnName]
         for pattern in patterns:
            if pattern.matches(entry):
               dirty = False
               break

         if dirty:
            newEntry = chooseRepair(data,patterns,entry)
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')




def getLLMPatterns(data, patterns):
   return None

def generateLLMCandidates(data: pandas.DataFrame, patterns, entry):
   return None

def generateCandidates(data: pandas.DataFrame, patterns, entry):
   dags = []
   for pattern in patterns:
      newDags = generateDAG(entry,pattern)
      dags.extend(newDags)
   candidates = []
   for dag in dags:
      moves,costs = generateMatrices(dag,entry)
      fillMatrices(dag,entry,moves,costs)
      candidates.extend()   
   return candidates

#This recursive methods takes an pattern and an entry and generates a Deterministic Acyclic Graph (short DAG)
#The Graph is stored as a list of lists, where each list contains one possible path through the DAG
#The inputs of the functions are the entry string and the pattern as a string and in reverse
def generateDAG(pattern: str, entry: str, Dag = [], index = 0):
   generated = []
   #print(pattern)
   char = pattern[index]

   #$ signals the end of a line in Regex
   #In our case it signals the start of the expression, since the expression should be given in reverse
   if char == '$':
      generated.extend(generateDAG(pattern,entry,Dag,index+1))

   #^ signals the start of a line in Regex
   #In our case it signals the end of the expression, since the expression should be given in reverse
   elif char == '^':
      generated.append(Dag)
      #print(f'^ generated: {generated}')

   #{x} in Regex means the previous statement is to be repeated exactly x times
   # The previous statement can be a group (X), a set [X], a special Sequence \X or a single character X
   elif char == '}':
      if not pattern[index+2] == '{':
         print(f"Given regular Expression {pattern} is not valid")
         exit(1)
      repeats = int(pattern[index+1])
      index += 2
      s,index = generateGroup(pattern,index)
      #print(s) 
      for i in range(0,repeats):
         temp = s.copy()
         temp.extend(Dag)
         Dag = temp
      #print(f'Dag after adding {s} exactly {repeats} times {Dag}')
      generated.extend(generateDAG(pattern,entry,Dag,index+1))
      #print(f'curvy ) generated: {generated}') 

   # + in Regex means the previous term can occur any number of times
   # The previous statement can be a group (X), a set [X], a special Sequence \X or a single character X   
   elif char == '*':
      s,index = generateGroup(pattern,index)
      bound = math.ceil(len(entry)/len(s))

      # Case where theres 0 appearences
      generated.extend(generateDAG(pattern,entry,Dag.copy(),index+1))

      #Other cases
      for i in range(0,bound):
         temp = s.copy()
         temp.extend(Dag)
         Dag = temp
         generated.extend(generateDAG(pattern,entry,Dag.copy(),index+1))
      #print(f'* generated: {generated}')

   # ? in Regex means the previous term can occur either 0 or 1 time at this place
   # The previous statement can be a group (X), a set [X], a special Sequence \X or a single character X
   elif char == '?':      
      s,index = generateGroup(pattern,index)
      bound = math.ceil(len(entry)/len(s))

      #Case where theres 0 appearences
      generated.extend(generateDAG(pattern,entry,Dag.copy(),index+1))

      #Case where theres 1 appearence
      temp = s.copy()
      temp.extend(Dag)
      Dag = temp
      generated.extend(generateDAG(pattern,entry,Dag.copy(),index+1))
      #print(f'? generated: {generated}')

   # + in Regex means the previous term can occur any number of times but must occur at least once
   # The previous statement can be a group (X), a set [X], a special Sequence \X or a single character X   
   elif char == '+':
      s,index = generateGroup(pattern,index)
      bound = math.ceil(len(entry)/len(s))
      for i in range(0,bound):
         temp = s.copy()
         temp.extend(Dag)
         Dag = temp
         generated.extend(generateDAG(pattern,entry,Dag.copy(),index+1))
      #print(f'+ generated: {generated}')

   #] in Regex signals the end of the set
   #The set is combined into one term and inserted into the DAG   
   elif char == ']':
      (s,index) = generateSet(pattern,index)
      Dag.insert(0,s)
      generated.extend(generateDAG(pattern,entry,Dag,index+1))
      #print(f'] generated: {generated}')

   #\ in Regex signals, that the following character is part of a Special Sequence
   #Since the \ stands before the character, we have to check for it one step earlier
   elif pattern[index+1] == '\\':
      Dag.insert(0,'\\' + char) 
      generated.extend(generateDAG(pattern,entry,Dag,index+1))
      #print(f'\\ generated: {generated}')       
   else:
      Dag.insert(0,char)
      generated.extend(generateDAG(pattern,entry,Dag,index+1))         
      #print(f'{char} generated: {generated}')   
   return generated

#This method is called when a repeating character (i.e +,*,?,{}) was found at pattern[index]
#These characters can be succeeded either by a set [X], a group (X), a special sequence \X or a character X
#For all of those cases this methods generates a list of the succeding terms.
#In every case but the group () this list will only have one entry
#The return values are the generated list and the index of the end of the succeding term

def generateGroup(pattern, index: int):
   s = []
   if pattern[index+1] == ']':
      (set,index) = generateSet(pattern,index+1)
      s.append(set)
   elif pattern[index+1] == ')':
      s = ''
   elif pattern[index+2] == '\\':
      s.append(pattern[index + 2] + pattern[index + 1])
      index += 2
   else:
      s.append(pattern[index+1])
      index += 1
   return s,index     

#This method is called, when a ] was detected at pattern[index]
#It scans the input for a matching [ later on in the input and saves everything in beetween into the string s
#The return values are a string s of type [X] and the index of the [ in the pattern
def generateSet(pattern, index: int):
   s = "]"
   while not pattern[index+1] == '[':
      s = s + pattern[index+1]
      index += 1
   s = s + '['
   return (s[::-1],index+1)

#The methods returns two matrices stored as dictionaries, both of size n times m, where n is the lenght of the given DAG and m is the length of the given entry string
# The cost matrix is filled with infinity at each key and the moves matrix is filled with the move None
def generateMatrices(Dag, entry: str):
   moves = {}
   costs = {}
   for i in range(len(Dag)+1):
      for j in range(len(entry)+1):
         costs[(j,i)] = math.inf
         moves[(j,i)] = None
   costs[(0,0)] = 0
   moves[(0,0)] = 'Start'     
   return moves,costs

#The method takes an DAG path, an entry string and two n times m matrices for costs and moves, as well as the index of the current entry.
#It tries to find the most efficient way to match the entry string to the DAG path by either matching a character of the entry string to an edge of the DAG,
#substituting a character with the value of an edge in the DAG, deleting a character or inserting the value of an edge in the DAG.
#These moves and their costs are stored in the two matrices costs and moves, where costs(i,j) is the minimal cost of getting to DAG edge j while using i characters of the string,
#with moves(i,j) being the last move to get this minimal cost 
def fillMatrices(Dag, entry: str, moves, costs, entryIndex = 0, DagIndex = 0 ):
   if entryIndex == len(entry) and DagIndex == len(Dag):
      return
   #Matching or Substituting a character
   if not entryIndex == len(entry) and not DagIndex == len(Dag):
      pattern = rf"^{Dag[DagIndex]}$"
      regex = re.compile(pattern)
      if costs[(entryIndex,DagIndex)] < costs[(entryIndex + 1,DagIndex + 1)]:
         #Character is matching
         if not regex.match(entry[entryIndex]) == None:
            costs[(entryIndex + 1,DagIndex + 1)] = costs[(entryIndex,DagIndex)]
            moves[(entryIndex + 1,DagIndex + 1)] = f"M"
            fillMatrices(Dag,entry,moves,costs,entryIndex + 1,DagIndex + 1)
         #Substituting the character   
         else:
            costs[(entryIndex + 1,DagIndex + 1)] = costs[(entryIndex,DagIndex)] + 1
            moves[(entryIndex + 1,DagIndex + 1)] = f"S({Dag[DagIndex]})"
            fillMatrices(Dag,entry,moves,costs,entryIndex + 1,DagIndex + 1)

   #Deleting a character
   if not entryIndex == len(entry) and costs[(entryIndex,DagIndex)] + 1 < costs[(entryIndex + 1,DagIndex)]:
      costs[(entryIndex + 1,DagIndex)] = costs[(entryIndex,DagIndex)] + 1
      moves[(entryIndex + 1,DagIndex)] = f"D({entry[entryIndex]})"
      fillMatrices(Dag,entry,moves,costs,entryIndex + 1,DagIndex)
   #Inserting an edge
   if not DagIndex == len(Dag) and costs[(entryIndex,DagIndex)] + 1 < costs[(entryIndex,DagIndex + 1)]:
      costs[(entryIndex,DagIndex + 1)] = costs[(entryIndex,DagIndex)] + 1
      moves[(entryIndex,DagIndex + 1)] = f"I({Dag[DagIndex]})"
      fillMatrices(Dag,entry,moves,costs,entryIndex,DagIndex + 1)

#readMovesMatrix takes a filled in moves matrix and recursively generates a candidate template using the shortest path given by the matrix.
#When called the inputs should be the moves matrix, the entry string and DAG used to generate the moves matrix, 
#the index of the bottom right entry of the moves matrix (len(entry), len(DAG)) and an empty list to store the candidate template into.
def readMovesMatrix(moves, entry: str, DAG, entryIndex: int, dagIndex: int, candidate = []):
   move = moves[(entryIndex,dagIndex)]
   if entryIndex < 0 or dagIndex < 0:
      print(f"The moves matrix was not constructed correctly! There can be no entry at key ({entryIndex},{dagIndex})")
      exit(1)
   if not entryIndex == 0 or not dagIndex == 0:
      if move[0] == 'M':
         temp = [entry[entryIndex - 1]]
         temp.extend(candidate)
         candidate = temp
         candidate = readMovesMatrix(moves,entry,DAG,entryIndex - 1,dagIndex - 1,candidate)
      elif move[0] == 'D':
         candidate = readMovesMatrix(moves,entry,DAG,entryIndex - 1,dagIndex,candidate)
      elif move[0] == 'I':
         temp = [DAG[dagIndex - 1]]
         temp.extend(candidate)
         candidate = temp
         candidate = readMovesMatrix(moves,entry,DAG,entryIndex,dagIndex - 1,candidate)
      else:
         temp = [DAG[dagIndex - 1]]
         temp.extend(candidate)
         candidate = temp
         candidate = readMovesMatrix(moves,entry,DAG,entryIndex - 1,dagIndex - 1,candidate) 
   return candidate              

   

def chooseRepair(data: pandas.DataFrame, patterns, entry: str):
   candidates = generateCandidates(data,patterns,entry)
   return 0


#The method learns patterns for the entries using Microsoft FlashProfile
#The patterns are then filtered by significance and the patterns only matching less than the Threshhold are discarded
def getPatterns(entries):
   try:
      patterns = learn_patterns(entries)
   except TypeError:
      print("Entries given to learn semantic patterns were not formatted as strings")   
   significantPatterns = []
   for pattern in patterns:
      if pattern.matching_fraction > SIGNIFICANCETHRESHHOLD:
         significantPatterns.append(pattern)
   return significantPatterns

#Prints a matrix stored as a dictionary with entry in row i at column j being given the key (i,j)
def printMatrix(matrix, numRows: int, numColumns: int):
   for i in range(numRows):
      s = f"{matrix[(i,0)]}"
      for j in range(1,numColumns):
         s += f" {matrix[(i,j)]}"
      print(s)   

if __name__ == "__main__":
    main()

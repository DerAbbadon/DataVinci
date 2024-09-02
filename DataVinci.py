import pandas
import math
import re
import numpy
import ollama
import os
import Levenshtein
import random
import time
import requests
import json
from sklearn import tree
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from pyprose.matching.text import learn_patterns
URL = "http://lyra.dbs.uni-hannover.de:11434/api/generate"
DATABASE_PATH = "./DGov_Typo/"
SIGNIFICANCE_THRESHHOLD = 0.1
SEMANTIC_TYPES = ["name", "country", "currency", "city", "year", "age", "ISBN", "day", "gender", "language", "nationality", "religion", "artist", "company", "industry", "species", "region", "address", "continent", "location"]

LLM_QUESTION = 'You are an expert in masking semantic types in data. You will be given a column of the database and are required to mask the data entries with different semantic types. Keep the following points in mind. 1. Mask semantic types in strings. 2. Only mask the semantic types specified: name, country, currency, city, year, age, ISBN, day, gender, language, nationality, religion, artist, company, industry, species, region, address, continent, location. 3. You are allowed to correct values, if the repaired value will be masked with a semantic type. 4. Include the whole Masked Column of the Task in your response and do not remove values, that do not need to be masked 5. You can mask multiple semantic types in a string, if there are multiple present 6.Do not include the examples in your response; <Examples> Data Column: Hannover-182, Hamburf-332, Berlin-252, Bremem-111, Muenchen-876; Masked Column: {city(Hannover)}-182, {city(Hamburg)}-332, {city(Berlin)}-252, {city(Bremen)}-111, {city(Muenchen)}-876; Data Column: Ind-674-PRO, US-823-JUN, US-237-JUN, Zim-843-PRO, Ind-473-JUN, usa_837, A2.A3.A4, A2.A3, AAA3, A5.A7., A8.A9., A3.A4.A5., A7., A6.A2.A3., A4.A3.A2.; Masked Column: {country(Ind)}-674-PRO, {country(US)}-823-JUN, {country(US)}-237-JUN, {country(Zim)}-843-PRO, {country(Ind)}-473-JUN, {country(US)}-837, A2.A3.A4, A2.A3, AAA3, A5.A7., A8.A9., A3.A4.A5., A7., A6.A2.A3., A4.A3.A2.; Data Column: John193, Sophie220, Mark188, Malt982, Maria309; Masked Column: {name(John)}193, {name(Sophie)}220, {name(Mark)}188, {name(Matt)}982, {name(Maria)}309; <Task> Data Column:'

QUESTION_END = '; Masked Column: '

#Weights for the heuristic ranker
levenshteinWeight = 0.25 #weigth of the Levenshtein distance beetween repaired value and original value
operationWeight = 0.25 #weigth of the count of alphanumeric edit operations used to generate repair value
closestNeighbourWeight = 0.3 #weigth of the minimal Levenshtein distance to the other values in the column
fractionWeight = 0.2 #weigth of the fraction of the column matching the pattern used to generate repair value


def main():
   calculateTotal()
   #iterateOverSubfoldersSkipping(DATABASE_PATH)
   db_in =pandas.read_csv("dirty.csv")
   db_in = db_in.astype(str)
   db = db_in.astype(str)
   #dirty = db_in.astype(str)
   '''
   examples = [
      "Ind-674-PRO",
      "US-823-JUN",
      "US-238-JUN",
      "Zim-843-PRO",
      "Eng-781-JUN",
      "Aus-664-PRO",
      "Ind-473-JUN",
      "Eng-573-JUN",
      "Zim-392-PRO",
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
      "A2.A3.A4.",
      "A2.A4.A6.",
      'Hannover-2024',
      'Hamburf-2023',
      'Berlin-2020',
      'Magdeburg-2021',
      'Berlin-2021',
      'Bremem-2019',
      'Muenchen-1997'
   ]
   patterns = ["^QUAL-[0-9]{2}$", "^[A-Z][a-z]+-[0-9]{3}-[A-Z]{3}$", "^A[0-9]\.A[0-9]\.A[0-9]\.$", "^A[0-9]\.A[0-9]\.$"]
   examples_short = [
      "pon{Tem-p}on\\34?3",
      "3A(3i)n.dree",
      "3[34p]et"]
   examples_masked= [
      "{country(Ind)}-674-{product(PRO)}",
      "{country(US)}-823-{season(JUN)}",
      "{country(US)}-238-{product(PRO)}",
      "{country(US)}_245",
      "{country(Zim)}-843-{season(JUN)}",
      "{country(Eng)}-573-{season(JUN)}",
      "{country(Zim)}-392-{product(PRO)}",
      "{city(Hamburg)}-2023",
      "{city(Berlin)}-2020",
      "{city(Magdeburg)}-2021",
      "{city(Berlin)}-2021",
      "{city(Bremen)}-2019",
      "{city(Muenchen)}-1997"
   ]
   '''
   #example_db = pandas.DataFrame(examples_short,columns=['examples'])
   #example_db =pandas.read_csv("example.csv").astype(str)
   #small_example_db = example_db['B'].to_frame()
   #columnMaskingWithLLM(db,'nlcsid')
   #columnMaskingWithLLMmultiplePrompts(db,'nlcsname',10)

   #DataVinci(db)
   #DataVinciNoLLM(db)
   #DataVinciNoLLMRandomForest(db)
   #DataVinciNoLLMSingleTree(db)

   #print(getMinimalEditDistance(example_db,'examples','Hamburg-2023'))

   #commonLengths = getCommonLengths(example_db,'examples')
   #stringConstants = generateStringConstantsOfColumn(example_db,'examples')
   #errormatrix = numpy.ones((3,1))
   #features = generateFeatureList(example_db,'examples',stringConstants,errormatrix,0,commonLengths)
   #print(f'examples: {example_db} \ncommonLengths: {commonLengths} \nstrings: {stringConstants} \nerrormatrix: {errormatrix} \nfeatures: {features}')

   #patterns = getPatterns(examples)
   #print(patterns)

   #pattern = '^[A-Z][a-z]+[\s]Trail$'
   #Dags = generateDAG(pattern[::-1],"{0E902E02-2119-4996-A1D0-07A62AD58607}")
   #print(Dags)

   #Dag = ['A','[0-9]','\\.','A','[0-9]','\\.']
   #entry = "AAA3"

   #Dag = ['[A-Z]', '[a-z]', '[a-z]', '[a-z]', '[a-z]', '[a-z]', '[a-z]', '[a-z]', '[a-z]', '[a-z]', '[\\s]', 'T', 'r', 'a', 'i', 'l']
   #entry = "CaliforniaTrail"
   #moves,costs = generateMatrices(Dag,entry)
   #fillMatrices(Dag,entry,moves,costs)
   #printMatrix(moves,len(entry) + 1, len(Dag) + 1)
   #printMatrix(costs,len(entry) + 1, len(Dag) + 1)
   
   #candidate = readMovesMatrix(moves,entry,Dag,len(entry), len(Dag))
   #candidate = handleSpecialRegex(candidate)
   #print(candidate)

   #clean =pandas.read_csv("clean.csv").astype(str)
   #TP,TN,FP,FN = calculateMetricsDetection(db,dirty,clean)
   #prec,rec,F1 = calculateF1Score(TP,TN,FP,FN)
   #print(f"Detection: TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, precision: {prec}, recall: {rec}, F1-score: {F1}")
   #TP,TN,FP,F1 = calculateMetricsRepair(db,dirty,clean)
   #prec,rec,F1 = calculateF1Score(TP,TN,FP,FN)
   #print(f"Repair: TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, precision: {prec}, recall: {rec}, F1-score: {F1}")
   



def DataVinci(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   for columnName in data.columns:
      columnMaskingWithLLM(data,columnName)
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("generated feature Dict")   
   columnIndex = 0         
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepair(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1  
   removeMasks(data)         


#########################################################################################################################################  
# Candidate Generation  #
#########################################################################################################################################

def generateCandidates(data: pandas.DataFrame, patterns, entry):
   candidates = []
   for pattern in patterns:
      newDags = generateDAG(str(pattern)[::-1],entry,[],0)
      print(f"Dags generated: pattern: {pattern}, Dags: {newDags}")
      for dag in newDags:
         moves,costs = generateMatrices(dag,entry)
         fillMatrices(dag,entry,moves,costs)
         candidate = readMovesMatrix(moves,entry,dag,len(entry), len(dag))
         candidate = handleSpecialRegex(candidate)
         #print(f"DAG: {dag} leads to candidate {candidate}")
         #printMatrix(moves,len(entry) + 1, len(dag) + 1)
         #printMatrix(costs,len(entry) + 1, len(dag) + 1)
         candidates.append((candidate,pattern,costs[(len(entry),len(dag))]))
      newDags = []      
   return candidates

#This recursive methods takes an pattern and an entry and generates a Deterministic Acyclic Graph (short DAG)
#The Graph is stored as a list of lists, where each list contains one possible path through the DAG
#The inputs of the functions are the entry string and the pattern as a string and in reverse
def generateDAG(pattern: str, entry: str, Dag = [], index = 0):
   generated = []
   #print(f"pattern: {pattern}, Dag: {Dag}")
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
      countindex = 1
      repeats = ''
      while index + countindex < len(pattern):
         if pattern[index+countindex] == '{':
            break
         repeats = pattern[index+countindex] + repeats
         countindex += 1
      #print(f'repeats: {repeats}')   
      if re.match('^[0-9]+$',repeats):   
         repeats = int(repeats)
         index += countindex
         s,index = generateGroup(pattern,index)
         #print(s) 
         for i in range(0,repeats):
            temp = s.copy()
            temp.extend(Dag)
            Dag = temp
         #print(f'Dag after adding {s} exactly {repeats} times {Dag}')
         generated.extend(generateDAG(pattern,entry,Dag,index+1))
         #print(f'curvy ) generated: {generated}')
      else:
         Dag.insert(0,'\\}')
         generated.extend(generateDAG(pattern,entry,Dag,index+1))

   # * in Regex means the previous term can occur any number of times
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
      Dag.insert(0,'\\'+ char) 
      generated.extend(generateDAG(pattern,entry,Dag,index+2))
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
      s.append('\\' + pattern[index + 1])
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
      regex = re.compile(re.escape(pattern))
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

def handleSpecialRegex(candidate):
   for i in range(0,len(candidate)):
      entry = candidate[i]
      if entry == '[\\s]': #Regex whitespace
         candidate[i] = ' '
      elif entry[0] == '\\':
         entry = re.sub('[\\\\]',"",entry)
         candidate[i] = entry      
   return candidate
                    

#################################################################################################################
#  Generating Repair based on candidate #
# ###############################################################################################################   

def chooseRepair(data: pandas.DataFrame, patterns, entry: str, rowIndex: int, columnIndex: int, columnName: str, featureDict: dict, errormatrix, stringConstants, commonLengths):
   candidates = generateCandidates(data,patterns,entry)
   rankedCandidates = {}
   candidatePatterns = {}
   print("candidates ready for concretizing")
   for value in candidates:
      (candidate,pattern,cost) = value
      abstractEditIndices = []
      for index in range(0,len(candidate)):
         if len(candidate[index]) > 1:
            abstractEditIndices.append(index)
      #print(f"candidate: {candidate} \n pattern: {pattern} \n abstract edit indices: {abstractEditIndices}")      
      if len(abstractEditIndices) > 0:
         encoder = LabelEncoder()      
         (featureVectors,labels) = chooseTrainingData(data, columnName, pattern, featureDict, abstractEditIndices)
         #print(f"training values: {featureVectors} \n labels: {labels}")
         encodedLabels = encoder.fit_transform(labels)
         xTrain, xTest, yTrain, yTest = train_test_split(featureVectors,encodedLabels,test_size=0.2)
         
         classifier = sampleDecisionTrees(xTest,xTrain,yTest,yTrain)
         print(f"candidate: {candidate} leads to classifier: {classifier}")
         candidateFeatures = assembleCandidateFeatures(data,rowIndex,columnIndex,candidate,featureDict,errormatrix,stringConstants,commonLengths)
         #print(f"candidate features: {candidateFeatures}, length of feature vector: {len(candidateFeatures)}")
         candidateLabel = classifier.predict(candidateFeatures)
         decodedLabel = encoder.inverse_transform(candidateLabel)
         #print(decodedLabel)
         concreteCandidate = concretizeCandidate(candidate,abstractEditIndices,decodedLabel[0])
         rankedCandidates[concreteCandidate] = cost
         candidatePatterns[concreteCandidate] = pattern
      else:
         stringCandidate = candidateToString(candidate)
         rankedCandidates[stringCandidate] = cost
         candidatePatterns[stringCandidate] = pattern
   heutisticRanker(data,columnName,rankedCandidates,entry,pattern)           
   sortedRankedCandidates = sorted(rankedCandidates, key=rankedCandidates.get, reverse=False)     
   return sortedRankedCandidates[0]

def chooseRepairSingleTree(data: pandas.DataFrame, patterns, entry: str, rowIndex: int, columnIndex: int, columnName: str, featureDict: dict, errormatrix, stringConstants, commonLengths):
   candidates = generateCandidates(data,patterns,entry)
   rankedCandidates = {}
   print("candidates ready for concretizing")
   for value in candidates:
      (candidate,pattern,cost) = value
      abstractEditIndices = []
      for index in range(0,len(candidate)):
         if len(candidate[index]) > 1:
            abstractEditIndices.append(index)
      #print(f"candidate: {candidate} \n pattern: {pattern} \n abstract edit indices: {abstractEditIndices}")      
      if len(abstractEditIndices) > 0:
         encoder = LabelEncoder()      
         (featureVectors,labels) = chooseTrainingData(data, columnName, pattern, featureDict, abstractEditIndices)
         #print(f"training values: {featureVectors} \n labels: {labels}")
         encodedLabels = encoder.fit_transform(labels)
         
         classifier = tree.DecisionTreeClassifier()
         classifier.fit(featureVectors,encodedLabels)
         print(f"candidate: {candidate} leads to classifier: {classifier}")
         candidateFeatures = assembleCandidateFeatures(data,rowIndex,columnIndex,candidate,featureDict,errormatrix,stringConstants,commonLengths)
         #print(f"candidate features: {candidateFeatures}, length of feature vector: {len(candidateFeatures)}")
         candidateLabel = classifier.predict(candidateFeatures)
         decodedLabel = encoder.inverse_transform(candidateLabel)
         #print(decodedLabel)
         concreteCandidate = concretizeCandidate(candidate,abstractEditIndices,decodedLabel[0])
         rankedCandidates[concreteCandidate] = cost
      else:
         stringCandidate = candidateToString(candidate)
         rankedCandidates[stringCandidate] = cost
   heutisticRanker(data,columnName,rankedCandidates,entry,pattern)           
   sortedRankedCandidates = sorted(rankedCandidates, key=rankedCandidates.get, reverse=False)     
   return sortedRankedCandidates[0]

def chooseRepairRandomForest(data: pandas.DataFrame, patterns, entry: str, rowIndex: int, columnIndex: int, columnName: str, featureDict: dict, errormatrix, stringConstants, commonLengths):
   candidates = generateCandidates(data,patterns,entry)
   rankedCandidates = {}
   print("candidates ready for concretizing")
   for value in candidates:
      (candidate,pattern,cost) = value
      abstractEditIndices = []
      for index in range(0,len(candidate)):
         if len(candidate[index]) > 1:
            abstractEditIndices.append(index)
      #print(f"candidate: {candidate} \n pattern: {pattern} \n abstract edit indices: {abstractEditIndices}")      
      if len(abstractEditIndices) > 0:
         encoder = LabelEncoder()      
         (featureVectors,labels) = chooseTrainingData(data, columnName, pattern, featureDict, abstractEditIndices)
         #print(f"training values: {featureVectors} \n labels: {labels}")
         encodedLabels = encoder.fit_transform(labels)
         
         classifier = RandomForestClassifier()
         classifier.fit(featureVectors,encodedLabels)
         print(f"candidate: {candidate} leads to classifier: {classifier.get_params()}")
         candidateFeatures = assembleCandidateFeatures(data,rowIndex,columnIndex,candidate,featureDict,errormatrix,stringConstants,commonLengths)
         #print(f"candidate features: {candidateFeatures}, length of feature vector: {len(candidateFeatures)}")
         candidateLabel = classifier.predict(candidateFeatures)
         decodedLabel = encoder.inverse_transform(candidateLabel)
         #print(decodedLabel)
         concreteCandidate = concretizeCandidate(candidate,abstractEditIndices,decodedLabel[0])
         rankedCandidates[concreteCandidate] = cost
      else:
         stringCandidate = candidateToString(candidate)
         rankedCandidates[stringCandidate] = cost
   heutisticRanker(data,columnName,rankedCandidates,entry,pattern)           
   sortedRankedCandidates = sorted(rankedCandidates, key=rankedCandidates.get, reverse=False)     
   return sortedRankedCandidates[0]

def generateStringConstantsOfColumn(data: pandas.DataFrame, columnName: str):
   constants = []
   column = data[columnName]
   for entry in column:
      #print(f"splitting entry {entry}")
      constants.append(entry)
      tempConstants = re.split('[^a-zA-z0-9]',str(entry))
      temp = []
      #print(f"tempContestants before splitting on caseChanges: {tempConstants}")
      for item in tempConstants: #Split on Case Changes
         if item == '': #Splitting on the non-alphanumeric values with regex can lead to empty strings, which we can filter at this line
            continue
         splits = []
         newItem = ""
         lastWasLower = False
         for c in item:
            if c.islower() == True:
               lastWasLower = True
            elif c.isupper() == True and lastWasLower == True:
               splits.append(newItem)
               newItem = ""
               lastWasLower = False
            else:
               lastWasLower = False
            newItem += c
         #Appending Last Item in the Loop   
         splits.append(newItem)   
         temp.extend(splits)
         #print(f"item {item} leads to splits {splits} and extends temp to {temp}")
      tempConstants = temp
      temp = []
      #print(f"tempContestants before splitting on continous alphabetic Changes: {tempConstants}")   
      for item in tempConstants: #Split on Changes between contiguous alphabetic and numeric characters, meaning at least 2 alphabetic characters followed by at least 2 numeric characters
         #print(f"splitting {item} on continuous alphabetic and numeric characters")
         splits = []
         newItem = ""
         tempNewItem = ""
         firstWasAlphabetic = False
         secondWasAlphabetic = False
         thirdWasNumeric = False
         for c in item:
            if c.isnumeric() and firstWasAlphabetic == True and secondWasAlphabetic == True and thirdWasNumeric == True:  
               splits.append(newItem)
               newItem = tempNewItem + c
               tempNewItem = ""
               firstWasAlphabetic = False
               secondWasAlphabetic = False
               thirdWasNumeric = False
            elif c.isnumeric() and firstWasAlphabetic == True and secondWasAlphabetic == True:
               thirdWasNumeric = True
               tempNewItem += c
            elif c.isalpha() and firstWasAlphabetic == True:
               secondWasAlphabetic = True
               newItem += c
            elif c.isalpha():
               firstWasAlphabetic = True
               newItem += c
            else:
               firstWasAlphabetic = False
               secondWasAlphabetic = False
               thirdWasNumeric = False
               newItem += tempNewItem + c
               tempNewItem = ""
         splits.append(newItem)   
         temp.extend(splits)
         #print(f"item {item} leads to splits {splits} and extends temp to {temp}")
      tempConstants = temp 
      temp = []

      #print(f"tempContestants before splitting on continous numeric Changes: {tempConstants}") 
      for item in tempConstants: #Split on Changes between contiguous numeric and alphabetic characters, meaning at least 2 numeric characters followed by at least 2 alphabetic characters
         #print(f"splitting {item} on continuous numeric and alphabetic characters")
         splits = []
         newItem = ""
         tempNewItem = ""
         firstWasNumeric  = False
         secondWasNumeric = False
         thirdWasAlphabetic = False
         for c in item:
            #print(f"character {c}, is alpha {c.isalpha()}, is numeric: {c.isnumeric()}")
            if c.isalpha() and firstWasNumeric  == True and secondWasNumeric  == True and thirdWasAlphabetic == True:  
               splits.append(newItem)
               newItem = tempNewItem + c
               tempNewItem = ""
               firstWasNumeric  = False
               secondWasNumeric  = False
               thirdWasAlphabetic = False
            elif c.isalpha() and firstWasNumeric  == True and secondWasNumeric  == True:
               thirdWasAlphabetic = True
               tempNewItem += c   
            elif c.isnumeric() and firstWasNumeric  == True:
               secondWasNumeric  = True
               newItem += c
            elif c.isnumeric():
               firstWasNumeric  = True
               newItem += c
            else:
               firstWasNumeric  = False
               secondWasNumeric  = False
               thirdWasAlphabetic = False
               newItem += tempNewItem + c
               tempNewItem = ""
         splits.append(newItem)   
         temp.extend(splits)
         #print(f"item {item} leads to splits {splits} and extends temp to {temp}")
      tempConstants = temp  
      constants.extend(tempConstants)
   constants = list(set(constants))                    
   return constants

def fillFeatureDict(data: pandas.DataFrame,featureDict: dict, stringConstants: dict, errormatrix, commonLengths: dict):
   for index,row in data.iterrows():
      columnIndex = 0
      for name in data.columns:
         featureDict[(index,columnIndex)] = generateFeaturVector(row[name],name,errormatrix,stringConstants[name],commonLengths[name],columnIndex,index)
         #print(f"feature in column {columnIndex} of length {len(featureDict[(index,columnIndex)])} added") 
         columnIndex += 1
         

def generateFeaturVector(entry: str, columnName: str, errormatrix, string_constants, commonLengths,columnIndex: int, rowIndex: int):
   featureVector = [hasDigits(entry)]
   featureVector.append(isNum(entry))
   featureVector.append(isError(columnIndex,rowIndex,errormatrix))
   featureVector.append(isNA(entry))
   featureVector.append(isText(entry))
   for length in commonLengths:
      featureVector.append(hasCommonLength(entry,length))
   for string in string_constants:
      featureVector.append(equals(entry,string))
      featureVector.append(contains(entry,string))
      featureVector.append(startsWith(entry,string))
      featureVector.append(endsWith(entry,string))
   return featureVector   

def heutisticRanker(data: pandas.DataFrame, columnName: str, repairCandidates: dict,  entry: str, pattern):
   editDistances = {}
   countOperations= {}
   minimalDistances = {}
   matchingFraction = pattern.matching_fraction
   for candidate in repairCandidates.keys():   
      editDistances[candidate] = Levenshtein.distance(candidate,entry)
      countOperations[candidate] = repairCandidates[candidate]
      minimalDistances[candidate] = getMinimalEditDistance(data,columnName,candidate)
   #Normalizing first 3 properties
   minEditDistance = min(editDistances.values())
   maxEditDistance = max(editDistances.values())
   minCount = min(countOperations.values())
   maxCount = max(countOperations.values())
   minMinDistance = min(minimalDistances.values())
   maxMinDistance = max(minimalDistances.values())
   #If all the values are the same the normalization (x-min)/(max-min) = 0/0, which gives an error, so we correct the max to min + 1 to get 0/1
   if minEditDistance == maxEditDistance:
      maxEditDistance += 1
   if minCount == maxCount:
      maxCount += 1
   if minMinDistance == maxMinDistance:
      maxMinDistance += 1      

   #print(f"dict: {editDistances} \n min: {minEditDistance}, max: {maxEditDistance}")
   for candidate in repairCandidates.keys():
      #print(f"weighted score: {levenshteinWeight} * ({editDistances[candidate]} - {minEditDistance}) / ({maxEditDistance} - {minEditDistance}) + {operationWeight} * ({countOperations[candidate]} - {minCount}) / ({maxCount} - {minCount}) + {closestNeighbourWeight} * ({minimalDistances[candidate]} - {minMinDistance}) / ({maxMinDistance} - {minMinDistance}) + {fractionWeight} * {matchingFraction}")   
      repairCandidates[candidate] = levenshteinWeight * (editDistances[candidate] - minEditDistance) / (maxEditDistance - minEditDistance) + operationWeight * (countOperations[candidate] - minCount) / (maxCount - minCount) + closestNeighbourWeight * (minimalDistances[candidate] - minMinDistance) / (maxMinDistance - minMinDistance) + fractionWeight * matchingFraction
      


def chooseTrainingData(data: pandas.DataFrame, columnName: str, pattern, featurDict: dict, abstractEditIndices):
   features = []
   labels = []
   for rowIndex,row in data.iterrows():
      entry = str(row[columnName])
      if pattern.matches(entry):
         label = ""
         for i in range(0,len(abstractEditIndices)):
            if i > 0:
               label += ","
            index =  abstractEditIndices[i]
            if index < len(entry):  
               label += entry[index]
            else: #The entry matches the pattern but is shorter than the candidate and wouldn't give enough information to generate a full repair candidate
               continue
         labels.append(label)      
         feature = []
         columnIndex = 0
         for column in data.columns:
            feature.extend(featurDict[(rowIndex,columnIndex)])
            columnIndex += 1
         features.append(feature)   
   #print(f"features: {features}, length of a feature: {len(features[0])}")                  
   trainingData = numpy.array(features)
   return (trainingData,labels)

def assembleCandidateFeatures(data: pandas.DataFrame, rowIndex: int, columnIndex: int, candidate, featureDict: dict, errormatrix,stringConstants,commonLengths):
   index = 0
   features = []
   stringCandidate = candidateToString(candidate)
   for columnName in data.columns:
      if index == columnIndex:
         features.extend(generateFeaturVector(stringCandidate,columnName,errormatrix,stringConstants,commonLengths, -1,-1))
      else:
         features.extend(featureDict[(rowIndex,index)])  
      index += 1
   feature_vector = numpy.array(features).reshape(1,-1)    
   return feature_vector

def concretizeCandidate(candidate, abstractIndizes,predictedLabel):
   labelIndex = 0
   predictedLabel = str(predictedLabel)
   if re.search(",",predictedLabel):
      labelList = re.split(",",predictedLabel)
   else:
      labelList = [predictedLabel]
   #print(f"predicted label: {predictedLabel}, with labelList {labelList} and abstractIndizes: {abstractIndizes}")      
   concreteCandidate = ""
   for index in range(0,len(candidate)):
      #print(f"candidate {concreteCandidate} at round {index}")
      if index in abstractIndizes:
         concreteCandidate += labelList[labelIndex]
         labelIndex += 1
      else:
         concreteCandidate += candidate[index]   
   return concreteCandidate

def sampleDecisionTrees(xTest,xTrain,yTest,yTrain):
   accuracyDict = {}

   #Decision tree without extra restraints always in the list
   unrestrictedDecisionTree = tree.DecisionTreeClassifier()
   unrestrictedDecisionTree.fit(xTrain,yTrain)
   unrestrictedDepth = unrestrictedDecisionTree.get_depth()
   unrestrictedNrNodes = unrestrictedDecisionTree.get_n_leaves()
   accuracyDict[unrestrictedDecisionTree] = unrestrictedDecisionTree.score(xTest,yTest)

   maxNodes = len(xTrain)
   maxDepth = math.ceil(math.log2(maxNodes))
   #print(f"max nodes: {maxNodes}, max depth: {maxDepth}")
   #print(f"Standard tree has: depth = {unrestrictedDepth}, nrNodes = {unrestrictedNrNodes}, accuracy = {accuracyDict[unrestrictedDecisionTree]}")
   depthsNodeCombos = []
   #A tree of depth 1 would just consist of two nodes, so its not very sensible. So we start at 2, with up to 4 classes. 
   for depth in range(2,maxDepth):
      upperBound = max(int(math.pow(2,depth)),maxNodes)
   #Only splitting one node at each depth level gives us the minimum number of leaf nodes for a tree of this depth, which is depth+1, since there is a leaf node at each level and at least 2 on the lowest level, compensating for the root node. Splitting each node at each depth level gives us the maximum number of nodes for a tree of this depth, which is 2^depth, since we start with 1 node at depth 0 and double from there.  Since the decisionTree can't have more than 1 leaf node per training sample, this number the maximum number of possible leaf nodes, so this number is used, if it's lower than the theoretical maximum 
      for numberNodes in range(depth+1,upperBound):
         depthsNodeCombos.append((numberNodes,depth))
   if len(depthsNodeCombos) > 1: #We have multiple decision trees to sample from
      bestTree = unrestrictedDecisionTree
      minDepth = unrestrictedDepth
      minNrNodes = unrestrictedNrNodes
      random.shuffle(depthsNodeCombos)
      nrTreesSampled = 1
      for nrNodes,depth in depthsNodeCombos: #Create Trees with the (max_nrNodes,max_depth) combos, which are possible
         decisionTree = tree.DecisionTreeClassifier(max_depth=depth,max_leaf_nodes=nrNodes)
         decisionTree.fit(xTrain,yTrain)
         accuracy = decisionTree.score(xTest,yTest)
         accuracyDict[decisionTree] = accuracy
         nrTreesSampled += 1
         if accuracy > 0.8:
            if decisionTree.get_depth() < minDepth:
                  bestTree = decisionTree
                  minDepth = bestTree.get_depth()
                  minNrNodes = bestTree.get_n_leaves()
            elif decisionTree.get_depth() == minDepth and decisionTree.get_n_leaves() < minNrNodes:
               bestTree = decisionTree
               minDepth = bestTree.get_depth()
               minNrNodes = bestTree.get_n_leaves()
         if nrTreesSampled >= 10: #we will sample 10 DecisionTrees
            break
      #print(accuracyDict)
      #print(f"Returned tree has: depth = {bestTree.get_depth()}, nrNodes = {bestTree.get_n_leaves()}, accuracy = {accuracyDict[bestTree]}, max number leaves = {bestTree.get_params()}")  
      return bestTree   
   else: #If there is only the unrestricted Tree, return this tree
      return unrestrictedDecisionTree   



##################################################################################################
#   Helper Functions #
# ################################################################################################

#The method learns patterns for the entries using Microsoft FlashProfile
#The patterns are then filtered by significance and the patterns only matching less than the Threshhold are discarded
def getPatterns(entries):
   try:
      patterns = learn_patterns(entries)
   except TypeError:
      print("Entries given to learn semantic patterns were not formatted as strings")   
   significantPatterns = []
   for pattern in patterns:
      if pattern.matching_fraction > SIGNIFICANCE_THRESHHOLD:
         significantPatterns.append(pattern)
   return significantPatterns

#Prints a matrix stored as a dictionary with entry in row i at column j being given the key (i,j)
def printMatrix(matrix, numRows: int, numColumns: int):
   for i in range(numRows):
      s = f"{matrix[(i,0)]}"
      for j in range(1,numColumns):
         s += f" {matrix[(i,j)]}"
      print(s)

# Calculates the precision, recall and F1_score of the repair based on the given result-, clean- and dirty-dataframe
def calculateMetricsRepair(result: pandas.DataFrame, dirty: pandas.DataFrame, clean: pandas.DataFrame):
   truePos,trueNeg,falsePos,falseNeg = 0,0,0,0
   for index,row in result.iterrows():
      for column in result.columns:
         #Entry was not changed correctly
         if result.at[index,column] == dirty.at[index,column] and result.at[index,column] == clean.at[index,column]:
            trueNeg += 1
         #Entry should have been changed but wasn't   
         elif result.at[index,column] == dirty.at[index,column] and not result.at[index,column] == clean.at[index,column]:
            falseNeg += 1
         #Entry was changed to the correct entry   
         elif not result.at[index,column] == dirty.at[index,column] and result.at[index,column] == clean.at[index,column]:
            truePos += 1
         #Entry was changed but not to the correct entry   
         elif not result.at[index,column] == dirty.at[index,column] and  not result.at[index,column] == clean.at[index,column]:
            falsePos += 1 
   return truePos,trueNeg,falsePos,falseNeg

def calculateF1Score(truePos: int, trueNeg: int, falsePos: int, falseNeg: int):
   if truePos == 0:
      precision,recall,F1_score = 0,0,0
   else:            
      precision =  truePos / (truePos + falsePos)
      recall = truePos / (truePos + falseNeg)
      F1_score = 2 * precision * recall / (precision + recall) 
   return precision,recall,F1_score  

# Calculates the precision, recall and F1_score of the detection based on the given result-, clean- and dirty-dataframe
def calculateMetricsDetection(result: pandas.DataFrame, dirty: pandas.DataFrame, clean: pandas.DataFrame):
   truePos,trueNeg,falsePos,falseNeg = 0,0,0,0
   for index,row in result.iterrows():
      for column in result.columns:
         #Entry was not changed correctly
         if result.at[index,column] == dirty.at[index,column] and dirty.at[index,column] == clean.at[index,column]:
            trueNeg += 1
         #Entry should have been changed but wasn't   
         elif result.at[index,column] == dirty.at[index,column] and not dirty.at[index,column] == clean.at[index,column]:
            falseNeg += 1
         #Entry was changed incorrecly   
         elif not result.at[index,column] == dirty.at[index,column] and dirty.at[index,column] == clean.at[index,column]:
            falsePos += 1
         #Entry was changed correcly  
         elif not result.at[index,column] == dirty.at[index,column] and  not dirty.at[index,column] == clean.at[index,column]:
            truePos += 1
   if truePos == 0:
      precision,recall,F1_score = 0,0,0
   else:            
      precision =  truePos / (truePos + falsePos)
      recall = truePos / (truePos + falseNeg)
      F1_score = 2 * precision * recall / (precision + recall)  
   return truePos,trueNeg,falsePos,falseNeg

#Returns the 5 most common lengths in the data column 
def getCommonLengths(data: pandas.DataFrame, columnName: str):
   column = data[columnName]              
   lengths = {} #dictionary with the lengths as keys and number of occurences as values
   for entry in column:
      entry = str(entry)
      if len(entry) in lengths.keys():
         lengths[len(entry)] += 1
      else:
         lengths[len(entry)] = 1      
   return sorted(lengths, key=lengths.get, reverse=True)[:5]

def equals(value: str, s: str):
   if value == s:
      return 1
   return 0

def contains(value: str, s: str):
   if re.search(rf"{re.escape(s)}",value):
      return 1
   return 0

def startsWith(value: str, s: str):
   if re.search(rf"^{re.escape(s)}",value):
      return 1
   return 0

def endsWith(value: str, s: str):
   if re.search(rf"{re.escape(s)}$",value):
      return 1
   return 0

def hasCommonLength(value: str, n: int):
   if len(value) == n:
      return 1
   return 0

def hasDigits(value: str):
   if re.search(rf"[0-9]",value):
      return 1
   return 0

def isNum(value: str):
   if re.match("^[0-9]+$",value):
      return 1
   return 0

def isError(columnindex: int, rowindex: int, errorMatrix):
   if columnindex == -1 and rowindex == -1: #Only used this way with a candidate, which should not be predicted as an error, so the feature isError is set to 0
      return 0
   return int(errorMatrix[rowindex][columnindex])

def isNA(value: str):
   if value == "" or value == " " or value == "NA" or value == "NaN":
      return 1
   return 0

def isText(value: str):
   if re.match("^[A-Za-z]+$",value):
      return 1
   return 0

def candidateToString(candidate):
   string = ""
   for i in range(0,len(candidate)):
      string += candidate[i]
   return string   

def columnMaskingWithLLM(data: pandas.DataFrame, columnName: str):
   dataShape = data.shape
   column = data[columnName].to_frame()
   columnString = ""
   for index,row in column.iterrows():
      if not index == 0:
         columnString += ", "
      entry = str(row[columnName])
      columnString += entry

   llmInput = LLM_QUESTION + columnString + QUESTION_END

   print(f"LLM input: {llmInput}")
   response = ollama.chat(model='llama3', messages=[
   {
      'role': 'user',
      'content': llmInput,
   },
   ])
   answerString = response['message']['content']
   print(f"LLM response: {response['message']['content']}")
   answerStringLines = re.split('\n',answerString)
   #print(f"LLM response split by linebreaks: {answerStringLines}")
   foundCandidate = False
   for line in answerStringLines:
      maskedValues = re.split(',',line)
      #print(f"{line} after splitting on ,: {maskedValues}; lenght = {len(maskedValues)} , data shape = {dataShape[0]}")
      if len(maskedValues) == dataShape[0]:
         if foundCandidate and not re.search('Masked Column:',line):
            print(f"found candidate already and {line} does not contain the key words Masked Column")
            continue
         for index,row in column.iterrows():
            repair = maskedValues[index]
            if index == 0: #First Value might have a prefix in front of it 
               repair = re.sub('Data Column: ',"",repair)
               repair = re.sub('Masked Column: ',"",repair)
            if index + 1 == dataShape[0]: #Last value might have an extra ";"
               repair = re.sub(';','',repair)
            re.sub('^[\s]*','',repair) # LLM might add whitespace after each ',' leading to each entry of the list, after the first, having extra whitespace at the front, which is removed here  
            data.at[index,columnName] = repair
         foundCandidate = True
   #print(f"masked data: {data}")
   #print(f"duration: {time.time() - startTime}")

def columnMaskingWithLlmMultiplePrompts(data: pandas.DataFrame, columnName: str, maxLength: int):
   startTime = time.time()
   column = data[columnName].to_frame()
   columnString = ""
   count = 0
   startIndex = 0
   for index,row in column.iterrows():
      if count == 0: #Index of the first value added to the columnString
         startIndex = index
      if not count == 0:
         columnString += ", "
      entry = str(row[columnName])
      columnString += entry
      count += 1
      if count == maxLength:
         llmInput = LLM_QUESTION + columnString + QUESTION_END

         #print(f"LLM input: {llmInput}")
         response = ollama.chat(model='llama3', messages=[
         {
            'role': 'user',
            'content': llmInput,
         },
         ])
         answerString = response['message']['content']
         #print(f"LLM response: {response['message']['content']}")
         answerStringLines = re.split('\n',answerString)
         #print(f"LLM response split by linebreaks: {answerStringLines}")
         foundCandidate = False
         for line in answerStringLines:
            maskedValues = re.split(',',line)
            #print(f"{line} after splitting on ,: {maskedValues}; lenght = {len(maskedValues)} , data shape = {count}")
            if len(maskedValues) == count:
               #print("replacing data")
               if foundCandidate and not re.search('Masked Column:',line):
                  print(f"found candidate already and {line} does not contain the key words Masked Column")
                  continue
               for i in range(0,count):
                  repair = maskedValues[i]
                  if i == 0: #First Value might have a prefix in front of it 
                     repair = re.sub('Data Column: ',"",repair)
                     repair = re.sub('Masked Column: ',"",repair)
                  if i + 1 == count: #Last value might have an extra ";"
                     repair = re.sub(';','',repair)
                  re.sub('[\s]*','',repair) # LLM might add whitespace after each ',' leading to each entry of the list, after the first, having extra whitespace at the front, which is removed here  
                  data.at[startIndex + i,columnName] = repair
               foundCandidate = True
         count = 0
         columnString = ""

   if count > 0: #LLM has to be prompted again with the last part of the column, since the number of cells in the column might not be divisible by 50
      llmInput = LLM_QUESTION + columnString + QUESTION_END

      #print(f"LLM input: {llmInput}")
      response = ollama.chat(model='llama3', messages=[
      {
         'role': 'user',
         'content': llmInput,
      },
      ])
      answerString = response['message']['content']
      #print(f"LLM response: {response['message']['content']}")
      answerStringLines = re.split('\n',answerString)
      #print(f"LLM response split by linebreaks: {answerStringLines}")
      foundCandidate = False
      for line in answerStringLines:
         maskedValues = re.split(',',line)
         #print(f"{line} after splitting on ,: {maskedValues}; \n lenght = {len(maskedValues)} , data shape = {count}")
         if len(maskedValues) == count:
            #print("replacing data")
            if foundCandidate and not re.search('Masked Column:',line):
               print(f"found candidate already and {line} does not contain the key words Masked Column")
               continue
            for i in range(0,count):
               repair = maskedValues[i]
               if i == 0: #First Value might have a prefix in front of it 
                  repair = re.sub('Data Column: ',"",repair)
                  repair = re.sub('Masked Column: ',"",repair)
               if i + 1 == count: #Last value might have an extra ";"
                  repair = re.sub(';','',repair)
               re.sub('[\s]*','',repair) # LLM might add whitespace after each ',' leading to each entry of the list, after the first, having extra whitespace at the front, which is removed here  
               data.at[startIndex + i,columnName] = repair
            foundCandidate = True       
   #print(f"masked data:")
   #for index,row in data.iterrows():
      #print(row[columnName])
   #print(f"duration: {time.time() - startTime}")          

def getMinimalEditDistance(data: pandas.DataFrame, columnName: str, entry: str):
   minDistance = -1
   column = data[columnName].to_frame()
   for index,row in column.iterrows():
      value = str(row[columnName])
      dist = Levenshtein.distance(entry,value)
      #print(f"entry: {entry}, value: {value}, minDist: {minDistance}, editDist: {dist}")
      if dist < minDistance or minDistance == -1:
         minDistance = dist
   return minDistance

def removeMasks(data: pandas.DataFrame):
   for index,row in data.iterrows():
      for columnName in data.columns:
         entry = row[columnName]
         match = re.search("\{[a-z]*\(.*\)\}",entry)
         #print(f"entry: {entry}, pattern found: {match}")
         if re.search("\{[a-z]*\(.*\)\}",entry): #Found a masked in entry
            for semantic in SEMANTIC_TYPES:
               regexString = "\{" + semantic + "\([A-Za-z0-9]*\)\}"         
               matches = re.findall(regexString,entry)
               #print(f"semantic type: {semantic} leads to matches {matches}")
               for match in matches:
                  withoutMask = re.sub("\{" + semantic + "\(","",match)
                  withoutMask = re.sub("\)\}","",withoutMask)
                  matchRegex = re.sub('\{','\{',match)
                  matchRegex = re.sub('\}','\}',matchRegex)
                  matchRegex = re.sub('\(','\(',matchRegex)
                  matchRegex = re.sub('\)','\)',matchRegex)
                  newEntry = re.sub(matchRegex,withoutMask,entry)
                  #print(f"match {match} with matchRegex {matchRegex} leads to unmasked value {withoutMask} and newEntry {newEntry}")
                  entry = newEntry
            data.at[(index,columnName)] = entry
   #print(data)           
            
def iterateOverSubfolders(directory: str):
   totalStartTime = time.time()
   saveName = "QuintetNo"
   try:
      number_df = pandas.read_csv(f"./{saveName}.csv")
      number_dict = number_df.to_dict('list')
      number_dict.pop('Unnamed: 0',None)
      print(number_dict)
   except FileNotFoundError:
      number_dict = {"folderName": [], "precisionDetection": [], "recallDetection": [], "fscoreDetection": [],"fireRate": [], "precisionRepair": [], "recallRepair": [], "fscoreRepair": [], "time": [], "tpDetection": [], "fpDetection": [], "fnDetection": [], "tnDetection": [],"tpRepair": [], "fpRepair": [], "fnRepair": [], "tnRepair": []}
   truePosDetection,trueNegDetection,falsePosDetection,falseNegDetection = 0,0,0,0
   truePosRepair,trueNegRepair,falsePosRepair,falseNegRepair = 0,0,0,0
   for path,folders,files in os.walk(directory):               
      for folder_name in folders:
         if folder_name in number_dict["folderName"]:
            print(f"skipping folder {folder_name} since its already in the dictionary")
            continue
         db_in =  pandas.read_csv(f"{path}/{folder_name}/dirty.csv")
         clean = pandas.read_csv(f"{path}/{folder_name}/clean.csv").astype(str)
         db_in.columns = clean.columns
         db = db_in.astype(str)
         dirty = db_in.astype(str)
         
         print(f"{folder_name}: \n dirty databas: {db}")
         startTime = time.time()

         #DataVinci(db)
         #DataVinciNoLLM(db)
         #DataVinciNoLLMRandomForest(db)
         #DataVinciNoLLMSingleTree(db)

         number_dict["folderName"].append(folder_name)
         number_dict["time"].append(time.time() - startTime)
         truePos,trueNeg,falsePos,falseNeg = calculateMetricsDetection(db,dirty,clean)
         precisionDetection,recallDetection,F1scoreDetection = calculateF1Score(truePos,trueNeg,falsePos,falseNeg)
         fireRate = (truePos + falsePos) / (truePos+trueNeg+falsePos+falseNeg)
         number_dict["precisionDetection"].append(precisionDetection)
         number_dict["recallDetection"].append(recallDetection)
         number_dict["fscoreDetection"].append(F1scoreDetection)
         number_dict["fireRate"].append(fireRate)
         number_dict["tpDetection"].append(truePos)
         number_dict["tnDetection"].append(trueNeg)
         number_dict["fpDetection"].append(falsePos)
         number_dict["fnDetection"].append(falseNeg)

         truePosDetection += truePos
         trueNegDetection += trueNeg
         falsePosDetection += falsePos
         falseNegDetection += falseNeg
         print(f"Detection current: TP {truePos}, TN {trueNeg}, FP {falsePos}, FN {falseNeg} \n total: TP {truePosDetection}, TN {trueNegDetection}, FP {falsePosDetection}, FN {falseNegDetection}")
         
         truePos,trueNeg,falsePos,falseNeg = calculateMetricsRepair(db,dirty,clean)
         precisionRepair,recallRepair,F1scoreRepair = calculateF1Score(truePos,trueNeg,falsePos,falseNeg)
         number_dict["precisionRepair"].append(precisionRepair)
         number_dict["recallRepair"].append(recallRepair)
         number_dict["fscoreRepair"].append(F1scoreRepair)
         number_dict["tpRepair"].append(truePos)
         number_dict["tnRepair"].append(trueNeg)
         number_dict["fpRepair"].append(falsePos)
         number_dict["fnRepair"].append(falseNeg)

         truePosRepair += truePos
         trueNegRepair += trueNeg
         falsePosRepair += falsePos
         falseNegRepair += falseNeg
         print(f"Repair current: TP {truePos}, TN {trueNeg}, FP {falsePos}, FN {falseNeg} \n total: TP {truePosRepair}, TN {trueNegRepair}, FP {falsePosRepair}, FN {falseNegRepair}")
         number_df = pandas.DataFrame(number_dict)
         number_df.to_csv(f"./{saveName}.csv",mode="w+")
   number_dict["folderName"].append("total")
   number_dict["time"].append(time.time() - totalStartTime)      
   precisionDetection,recallDetection,F1scoreDetection = calculateF1Score(truePosDetection,trueNegDetection,falsePosDetection,falseNegDetection)
   fireRate = (truePosDetection + falsePosDetection) / (truePosDetection+trueNegDetection+falsePosDetection+falseNegDetection)
   precisionRepair,recallRepair,F1scoreRepair = calculateF1Score(truePosRepair,trueNegRepair,falsePosRepair,falseNegRepair)
   print(f"Detection: precision: {precisionDetection}, recall: {recallDetection}, F1-score: {F1scoreDetection}, fire rate: {fireRate} \nRepair: precision: {precisionRepair}, recall: {recallRepair}, F1-score: {F1scoreRepair}")
   number_dict["precisionDetection"].append(precisionDetection)
   number_dict["recallDetection"].append(recallDetection)
   number_dict["fscoreDetection"].append(F1scoreDetection)
   number_dict["fireRate"].append(fireRate)
   number_dict["precisionRepair"].append(precisionRepair)
   number_dict["recallRepair"].append(recallRepair)
   number_dict["fscoreRepair"].append(F1scoreRepair)
   number_dict["tpDetection"].append(truePosDetection)
   number_dict["tnDetection"].append(trueNegDetection)
   number_dict["fpDetection"].append(falsePosDetection)
   number_dict["fnDetection"].append(falseNegDetection)
   number_dict["tpRepair"].append(truePosRepair)
   number_dict["tnRepair"].append(trueNegRepair)
   number_dict["fpRepair"].append(falsePosRepair)
   number_dict["fnRepair"].append(falseNegRepair)
   number_df = pandas.DataFrame(number_dict)
   number_df.to_csv(f"./{saveName}.csv")


################################################################################################################################################################
# DataVinci Variants
################################################################################################################################################################   

def DataVinciNoLLM(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")   
   print("feature dict constructed")
   columnIndex = 0         
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepair(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1

def DataVinciMultipleLLMPrompts(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   for columnName in data.columns:
      columnMaskingWithLlmMultiplePrompts(data,columnName,10)
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("feature dict constructed")   
   columnIndex = 0         
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepair(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1      

def DataVinciNoLLMSingleTree(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("feature dict constructed")   
   columnIndex = 0         
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepairSingleTree(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1          

def DataVinciNoLLMRandomForest(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("feature dict constructed")   
   columnIndex = 0         
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepairRandomForest(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1 

def iterateOverSubfoldersSkipping(directory: str):
   totalStartTime = time.time()
   saveName = "DGovSkippingLLM"
   #saveName = "DGovSkippingSingleTree"
   #saveName = "DGovSkippingRandomForest"
   try:
      number_df = pandas.read_csv(f"./{saveName}.csv")
      number_dict = number_df.to_dict('list')
      number_dict.pop('Unnamed: 0',None)
      print(number_dict)
   except FileNotFoundError:
      number_dict = {"folderName": [], "precisionDetection": [], "recallDetection": [], "fscoreDetection": [],"fireRate": [], "precisionRepair": [], "recallRepair": [], "fscoreRepair": [], "time": [], "tpDetection": [], "fpDetection": [], "fnDetection": [], "tnDetection": [],"tpRepair": [], "fpRepair": [], "fnRepair": [], "tnRepair": [], "skippedColumns": []}
   truePosDetection,trueNegDetection,falsePosDetection,falseNegDetection = 0,0,0,0
   truePosRepair,trueNegRepair,falsePosRepair,falseNegRepair = 0,0,0,0
   for path,folders,files in os.walk(directory):               
      for folder_name in folders:
         if folder_name in number_dict["folderName"]:
            print(f"skipping folder {folder_name} since its already in the dictionary")
            continue
         db_in =  pandas.read_csv(f"{path}/{folder_name}/dirty.csv")
         clean = pandas.read_csv(f"{path}/{folder_name}/clean.csv").astype(str)
         db_in.columns = clean.columns
         db = db_in.astype(str)
         dirty = db_in.astype(str)
         if db.shape[0] > 150:
            print(f"skipping {folder_name}")
            number_dict["folderName"].append(folder_name)
            number_dict["time"].append(0)
            number_dict["precisionDetection"].append(0)
            number_dict["recallDetection"].append(0)
            number_dict["fscoreDetection"].append(0)
            number_dict["fireRate"].append(0)
            number_dict["tpDetection"].append(0)
            number_dict["tnDetection"].append(0)
            number_dict["fpDetection"].append(0)
            number_dict["fnDetection"].append(0)
            number_dict["precisionRepair"].append(0)
            number_dict["recallRepair"].append(0)
            number_dict["fscoreRepair"].append(0)
            number_dict["tpRepair"].append(0)
            number_dict["tnRepair"].append(0)
            number_dict["fpRepair"].append(0)
            number_dict["fnRepair"].append(0)
            number_dict["skippedColumns"].append('')
            continue
         print(f"{folder_name}: \n dirty databas: {db}")
         startTime = time.time()

         skippedColumns = DataVinciSkipping(db)
         #skippedColumns = DataVinciSkippingNoLLM(db)
         #skippedColumns = DataVinciSkippingNoLLMRandomForest(db)
         #skippedColumns = DataVinciSkippingNoLLMSingleTree(db)

         number_dict["folderName"].append(folder_name)
         number_dict["time"].append(time.time() - startTime)
         truePos,trueNeg,falsePos,falseNeg = calculateMetricsDetectionSkipping(db,dirty,clean,skippedColumns)
         precisionDetection,recallDetection,F1scoreDetection = calculateF1Score(truePos,trueNeg,falsePos,falseNeg)
         fireRate = (truePos + falsePos) / (truePos+trueNeg+falsePos+falseNeg)
         number_dict["precisionDetection"].append(precisionDetection)
         number_dict["recallDetection"].append(recallDetection)
         number_dict["fscoreDetection"].append(F1scoreDetection)
         number_dict["fireRate"].append(fireRate)
         number_dict["tpDetection"].append(truePos)
         number_dict["tnDetection"].append(trueNeg)
         number_dict["fpDetection"].append(falsePos)
         number_dict["fnDetection"].append(falseNeg)

         truePosDetection += truePos
         trueNegDetection += trueNeg
         falsePosDetection += falsePos
         falseNegDetection += falseNeg
         print(f"Detection current: TP {truePos}, TN {trueNeg}, FP {falsePos}, FN {falseNeg} \n total: TP {truePosDetection}, TN {trueNegDetection}, FP {falsePosDetection}, FN {falseNegDetection}")
         
         truePos,trueNeg,falsePos,falseNeg = calculateMetricsRepairSkipping(db,dirty,clean,skippedColumns)
         precisionRepair,recallRepair,F1scoreRepair = calculateF1Score(truePos,trueNeg,falsePos,falseNeg)
         number_dict["precisionRepair"].append(precisionRepair)
         number_dict["recallRepair"].append(recallRepair)
         number_dict["fscoreRepair"].append(F1scoreRepair)
         number_dict["tpRepair"].append(truePos)
         number_dict["tnRepair"].append(trueNeg)
         number_dict["fpRepair"].append(falsePos)
         number_dict["fnRepair"].append(falseNeg)

         truePosRepair += truePos
         trueNegRepair += trueNeg
         falsePosRepair += falsePos
         falseNegRepair += falseNeg
         skipped = ''
         if len(skippedColumns) > 0:
            skipped = skippedColumns[0]
            for i in range(1,len(skippedColumns)):
               skipped +=  ';' + skippedColumns[i]  
         number_dict["skippedColumns"].append(skipped)

         print(f"Repair current: TP {truePos}, TN {trueNeg}, FP {falsePos}, FN {falseNeg} \n total: TP {truePosRepair}, TN {trueNegRepair}, FP {falsePosRepair}, FN {falseNegRepair}")
         number_df = pandas.DataFrame(number_dict)
         number_df.to_csv(f"./{saveName}.csv",mode="w+")
   number_dict["folderName"].append("total")
   number_dict["time"].append(time.time() - totalStartTime)      
   precisionDetection,recallDetection,F1scoreDetection = calculateF1Score(truePosDetection,trueNegDetection,falsePosDetection,falseNegDetection)
   fireRate = (truePosDetection + falsePosDetection) / (truePosDetection+trueNegDetection+falsePosDetection+falseNegDetection)
   precisionRepair,recallRepair,F1scoreRepair = calculateF1Score(truePosRepair,trueNegRepair,falsePosRepair,falseNegRepair)
   print(f"Detection: precision: {precisionDetection}, recall: {recallDetection}, F1-score: {F1scoreDetection}, fire rate: {fireRate} \nRepair: precision: {precisionRepair}, recall: {recallRepair}, F1-score: {F1scoreRepair}")
   number_dict["precisionDetection"].append(precisionDetection)
   number_dict["recallDetection"].append(recallDetection)
   number_dict["fscoreDetection"].append(F1scoreDetection)
   number_dict["fireRate"].append(fireRate)
   number_dict["precisionRepair"].append(precisionRepair)
   number_dict["recallRepair"].append(recallRepair)
   number_dict["fscoreRepair"].append(F1scoreRepair)
   number_dict["tpDetection"].append(truePosDetection)
   number_dict["tnDetection"].append(trueNegDetection)
   number_dict["fpDetection"].append(falsePosDetection)
   number_dict["fnDetection"].append(falseNegDetection)
   number_dict["tpRepair"].append(truePosRepair)
   number_dict["tnRepair"].append(trueNegRepair)
   number_dict["fpRepair"].append(falsePosRepair)
   number_dict["fnRepair"].append(falseNegRepair)
   number_dict["skippedColumns"].append('')
   number_df = pandas.DataFrame(number_dict)
   number_df.to_csv(f"./{saveName}.csv")

def DataVinciSkipping(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   skippedColumns = []
   for columnName in data.columns:
      columnMaskingWithLLM(data,columnName)
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      for i in range(0,len(commonLengths[columnName])):
         if commonLengths[columnName][i] > 15:
            skippedColumns.append(columnName)
            break      
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("generated feature Dict")   
   columnIndex = 0         
   for columnName in data.columns:
      if columnName in skippedColumns:
         print(f"skipping column {columnName} for too long entries")
         columnIndex += 1
         continue
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepair(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1  
   removeMasks(data)
   return skippedColumns

def DataVinciSkippingNoLLM(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   skippedColumns = []
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      for i in range(0,len(commonLengths[columnName])):
         if commonLengths[columnName][i] > 15:
            skippedColumns.append(columnName)
            break      
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("generated feature Dict")   
   columnIndex = 0         
   for columnName in data.columns:
      if columnName in skippedColumns:
         print(f"skipping column {columnName} for too long entries")
         columnIndex += 1
         continue
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepair(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1  
   return skippedColumns

def DataVinciSkippingNoLLMSingleTree(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   skippedColumns = []
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      for i in range(0,len(commonLengths[columnName])):
         if commonLengths[columnName][i] > 15:
            skippedColumns.append(columnName)
            break      
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("generated feature Dict")   
   columnIndex = 0         
   for columnName in data.columns:
      if columnName in skippedColumns:
         print(f"skipping column {columnName} for too long entries")
         columnIndex += 1
         continue
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepairSingleTree(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1  
   return skippedColumns

def DataVinciSkippingNoLLMRandomForest(data: pandas.DataFrame):
   columnIndex = 0
   errorMatrix = numpy.ones(data.shape)
   stringConstants = {}
   commonLengths = {}
   featureDict = {}
   patterns = {}
   skippedColumns = []
   for columnName in data.columns:
      column = data[columnName]
      columnDF = column.to_frame()
      patterns[columnName] = getPatterns(column)
      stringConstants[columnName] = generateStringConstantsOfColumn(data,columnName)
      commonLengths[columnName] = getCommonLengths(data,columnName)
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         for pattern in patterns[columnName]:
            if pattern.matches(entry):
               errorMatrix[index, columnIndex] = 0
               break
      for i in range(0,len(commonLengths[columnName])):
         if commonLengths[columnName][i] > 15:
            skippedColumns.append(columnName)
            break      
      columnIndex += 1
   print(f"patterns: {patterns}")
   #print(f"stringConstants: {stringConstants} ,commonLengths: {commonLengths}")   
   fillFeatureDict(data,featureDict,stringConstants,errorMatrix,commonLengths)
   #print(f"feature dict: {featureDict}")
   print("generated feature Dict")   
   columnIndex = 0         
   for columnName in data.columns:
      if columnName in skippedColumns:
         print(f"skipping column {columnName} for too long entries")
         columnIndex += 1
         continue
      column = data[columnName]
      columnDF = column.to_frame()         
      for index,row in columnDF.iterrows():
         entry = row[columnName]
         if errorMatrix[index,columnIndex] == 1:
            newEntry = chooseRepairRandomForest(data,patterns[columnName],entry,index,columnIndex,columnName,featureDict,errorMatrix,stringConstants[columnName],commonLengths[columnName])
            data.at[index,columnName] = newEntry
            print(f'found dirty entry {entry} at row {index} in column {columnName} and replaced it with {data.at[index,columnName]}')
      columnIndex += 1  
   return skippedColumns

def columnMaskingWithLLMRequests(data: pandas.DataFrame, columnName: str):
   dataShape = data.shape
   column = data[columnName].to_frame()
   columnString = ""
   for index,row in column.iterrows():
      if not index == 0:
         columnString += ", "
      entry = str(row[columnName])
      columnString += entry

   llmInput = LLM_QUESTION + columnString + QUESTION_END

   inputData = {
   "model": 
   "llama3",
   "prompt": 
   llmInput,
   }

   response = requests.post(URL, data=json.dumps(inputData))
   print(response)
   answerString = response['message']['content']
   print(f"LLM response: {response['message']['content']}")
   answerStringLines = re.split('\n',answerString)
   #print(f"LLM response split by linebreaks: {answerStringLines}")
   for line in answerStringLines:
      maskedValues = re.split(',',line)
      #print(f"{line} after splitting on ,: {maskedValues}; lenght = {len(maskedValues)} , data shape = {dataShape[0]}")
      if len(maskedValues) == dataShape[0]:
         #print("replacing data")
         for index,row in column.iterrows():
            repair = maskedValues[index]
            if index == 0: #First Value might have a prefix in front of it 
               repair = re.sub('Data Column: ',"",repair)
               repair = re.sub('Masked Column: ',"",repair)
            if index + 1 == dataShape[0]: #Last value might have an extra ";"
               repair = re.sub(';','',repair)
            re.sub('^[\s]*','',repair) # LLM might add whitespace after each ',' leading to each entry of the list, after the first, having extra whitespace at the front, which is removed here  
            data.at[index,columnName] = repair
         break
   #print(f"masked data: {data}")

   # Calculates the precision, recall and F1_score of the repair based on the given result-, clean- and dirty-dataframe
def calculateMetricsRepairSkipping(result: pandas.DataFrame, dirty: pandas.DataFrame, clean: pandas.DataFrame,skippedColumns):
   truePos,trueNeg,falsePos,falseNeg = 0,0,0,0
   for index,row in result.iterrows():
      for column in result.columns:
         if column in skippedColumns:
            continue
         #Entry was not changed correctly
         if result.at[index,column] == dirty.at[index,column] and result.at[index,column] == clean.at[index,column]:
            trueNeg += 1
         #Entry should have been changed but wasn't   
         elif result.at[index,column] == dirty.at[index,column] and not result.at[index,column] == clean.at[index,column]:
            falseNeg += 1
         #Entry was changed to the correct entry   
         elif not result.at[index,column] == dirty.at[index,column] and result.at[index,column] == clean.at[index,column]:
            truePos += 1
         #Entry was changed but not to the correct entry   
         elif not result.at[index,column] == dirty.at[index,column] and  not result.at[index,column] == clean.at[index,column]:
            falsePos += 1 
   return truePos,trueNeg,falsePos,falseNeg

# Calculates the precision, recall and F1_score of the detection based on the given result-, clean- and dirty-dataframe
def calculateMetricsDetectionSkipping(result: pandas.DataFrame, dirty: pandas.DataFrame, clean: pandas.DataFrame,skippedColumns):
   truePos,trueNeg,falsePos,falseNeg = 0,0,0,0
   for index,row in result.iterrows():
      for column in result.columns:
         if column in skippedColumns:
            continue
         #Entry was not changed correctly
         if result.at[index,column] == dirty.at[index,column] and dirty.at[index,column] == clean.at[index,column]:
            trueNeg += 1
         #Entry should have been changed but wasn't   
         elif result.at[index,column] == dirty.at[index,column] and not dirty.at[index,column] == clean.at[index,column]:
            falseNeg += 1
         #Entry was changed incorrecly   
         elif not result.at[index,column] == dirty.at[index,column] and dirty.at[index,column] == clean.at[index,column]:
            falsePos += 1
         #Entry was changed correcly  
         elif not result.at[index,column] == dirty.at[index,column] and  not dirty.at[index,column] == clean.at[index,column]:
            truePos += 1
   if truePos == 0:
      precision,recall,F1_score = 0,0,0
   else:            
      precision =  truePos / (truePos + falsePos)
      recall = truePos / (truePos + falseNeg)
      F1_score = 2 * precision * recall / (precision + recall)  
   return truePos,trueNeg,falsePos,falseNeg

def calculateTotal():
   #saveName = "fRahaDgov9"
   #saveName = "BaranRahaDgov0"
   #saveName = "DGovSkippingLLM"
   saveName = "DGovSkippingSingleTree"
   #saveName = "DGovSkippingRandomForest"
   try:
      number_df = pandas.read_csv(f"./{saveName}.csv")
      number_dict = number_df.to_dict('list')
      number_dict.pop('Unnamed: 0',None)
      print(number_dict)
   except FileNotFoundError:
      number_dict = {"folderName": [], "precisionDetection": [], "recallDetection": [], "fscoreDetection": [],"fireRate": [], "precisionRepair": [], "recallRepair": [], "fscoreRepair": [], "time": [], "tpDetection": [], "fpDetection": [], "fnDetection": [], "tnDetection": [],"tpRepair": [], "fpRepair": [], "fnRepair": [], "tnRepair": [], "skippedColumns": []}
   truePosDetection,trueNegDetection,falsePosDetection,falseNegDetection = 0,0,0,0
   truePosRepair,trueNegRepair,falsePosRepair,falseNegRepair = 0,0,0,0
   totalTime = 0
   for i in range(0,len(number_dict["folderName"])):
         folder = number_dict["folderName"][i]
         print(f"processing folder {folder}")
         truePosDetection += number_dict["tpDetection"][i]
         trueNegDetection += number_dict["tnDetection"][i]
         falsePosDetection += number_dict["fpDetection"][i]
         falseNegDetection += number_dict["fnDetection"][i]

         truePosRepair += number_dict["tpRepair"][i]
         trueNegRepair += number_dict["tnRepair"][i]
         falsePosRepair += number_dict["fpRepair"][i]
         falseNegRepair += number_dict["fnRepair"][i]

         totalTime += number_dict["time"][i]
   number_dict["folderName"].append("total")
   number_dict["time"].append(totalTime)      
   precisionDetection,recallDetection,F1scoreDetection = calculateF1Score(truePosDetection,trueNegDetection,falsePosDetection,falseNegDetection)
   fireRate = (truePosDetection + falsePosDetection) / (truePosDetection+trueNegDetection+falsePosDetection+falseNegDetection)
   precisionRepair,recallRepair,F1scoreRepair = calculateF1Score(truePosRepair,trueNegRepair,falsePosRepair,falseNegRepair)
   print("DataVinci performance on {}:\nDetection:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}\nFireRate = {:.2f}\nRepair:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}\nTime = {}".format(saveName, precisionDetection, recallDetection, F1scoreDetection,fireRate,precisionRepair,recallRepair,F1scoreRepair,totalTime))
   print("DataVinci performance on {}:\nDetection:\nTrue Positive = {:.2f}\nFalse Positive = {:.2f}\nTrue Negatice= {:.2f}\nFalse Negatice= {:.2f}\nRepair:\nTrue Positive = {:.2f}\nFalse Positive = {:.2f}\nTrue Negatice= {:.2f}\nFalse Negatice= {:.2f}".format(saveName, truePosDetection, falsePosDetection,trueNegDetection,falseNegDetection,truePosRepair,falsePosRepair,trueNegRepair,falseNegRepair))
   number_dict["precisionDetection"].append(precisionDetection)
   number_dict["recallDetection"].append(recallDetection)
   number_dict["fscoreDetection"].append(F1scoreDetection)
   number_dict["fireRate"].append(fireRate)
   number_dict["precisionRepair"].append(precisionRepair)
   number_dict["recallRepair"].append(recallRepair)
   number_dict["fscoreRepair"].append(F1scoreRepair)
   number_dict["tpDetection"].append(truePosDetection)
   number_dict["tnDetection"].append(trueNegDetection)
   number_dict["fpDetection"].append(falsePosDetection)
   number_dict["fnDetection"].append(falseNegDetection)
   number_dict["tpRepair"].append(truePosRepair)
   number_dict["tnRepair"].append(trueNegRepair)
   number_dict["fpRepair"].append(falsePosRepair)
   number_dict["fnRepair"].append(falseNegRepair)
   number_dict["skippedColumns"].append('')
   number_df = pandas.DataFrame(number_dict)
   number_df.to_csv(f"./{saveName}.csv")

if __name__ == "__main__":
   main()


def mountDrive():
  from google.colab import drive
  drive.mount('/content/gdrive')

def getDatasetFromGDrive(fileName):
  import os
  os.system(''.join(['cp "/content/gdrive/My Drive/',fileName,'"']))

def getPreviousWeightsFromGDrive(weightFolderName):
  import os
  os.system(''.join(['cp -r "/content/gdrive/My Drive/',weightFolderName,'" .']))

def getLatestModelName(weightFolderName):
  import os
  all_files = [(name, os.path.getmtime(''.join(['.',weightFolderName,'/',name]))) for name in os.listdir(''.join(['.',weightFolderName,'/']))]
  latest_uploaded_file = sorted(all_files, key=lambda x: -x[1])[0][0]
  return './MLP_Weights/'+latest_uploaded_file

def readDataset(fileName):
  with open(fileName,'r') as f:
    txt = f.read()
  return txt

def getListFromChar(char, chardict, classes):
  num = chardict[char]
  return [0]*num+[1]+[0]*(classes-1-num)

def getChars():
  import string
  chars = string.ascii_lowercase+'.," '
  return chars

def getCharacterVars(indexIn, chars=None):
  if chars is None:
    chars = getChars()
  classes = len(chars)
  charDict = {chars[i]: i for i in range(len(chars))}
  if indexIn:
    charDictList = {chars[i]: 1-i/classes/2 for i in range(classes)}
  else:
    charDictList = {chars[i]: getListFromChar(chars[i], charDict, classes) for i in range(classes)}
  return chars, charDict, charDictList, classes

def reformatString(inputString, chars):
  """
  WARNING: This function removes all characters it has no clue about.
  This includes anything related to numbers as well as most symbols.
  If those characters are required in any way, do some preproprecessing,
  such as replacing '1' with 'one'.
  """
  inputString = ' '.join(inputString.split())
  inputString = inputString.lower()
  inputString = ''.join([c for c in inputString if c in chars])
  return inputString

def getNeuronList(neuronsPerLayer, layer, classNeurons, classes):
  if classNeurons:
    neuronsPerLayer *= classes
  return [neuronsPerLayer]*layer
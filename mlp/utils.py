def mountDrive():
  from google.colab import drive
  drive.mount('/content/gdrive')

def getDatasetFromGDrive(fileName):
  import os
  os.system(''.join(['cp "/content/gdrive/My Drive/',fileName,'" .']))

def getPreviousWeightsFromGDrive(weightFolderName):
  import os
  os.system(''.join(['cp -r "/content/gdrive/My Drive/',weightFolderName,'" .']))

def getLatestModelName(weightFolderName):
  import os
  all_files = [(name, os.path.getmtime(''.join(['./',weightFolderName,'/',name]))) for name in os.listdir(''.join(['./',weightFolderName,'/']))]
  latest_uploaded_file = sorted(all_files, key=lambda x: -x[1])[0][0]
  return ''.join(['./',weightFolderName,'/',latest_uploaded_file])

def readDataset(fileName):
  with open(fileName,'r',errors='ignore') as f:
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

def getTestString(charSet=None):
  testString = """You then insert the tool into the holes evident on the blinkie's surface.  The
location of the hole depends on the color of the blinkie (and manufacturer).
The blue blinkie commonly found in wealthy suburban areas is disabled by
inserting the tool into the hole directly in the middle of the blinkie's body.
You should hear a "click" and the blinkie will cease to function.  The red
blinkie commonly found in highway construction sights uses the offset hole
located in either the upper hand right or left of the blinkie's body.  You will
NOT hear a click with the red one because it uses a slide switch instead of a
pushbutton one.  Again, the blinkie will turn off.  Yellow and black blinkies
turn off in a similar way as the red ones."""
  if charSet is None:
    charSet = getChars()
  testString = reformatString(testString, charSet)
  return testString

def getTfGenerator(pythonGenerator, batchSize, outputs):
  import tensorflow as tf
  if outputs > 1:
    tfGenerator = tf.data.Dataset.from_generator(generator=lambda: map(tuple, pythonGenerator),
                                                 output_types=(tf.float32,tf.float32),
                                                 output_shapes=(tf.TensorShape((None,)), tf.TensorShape((outputs,1)))
                                                )
  else:
    tfGenerator = tf.data.Dataset.from_generator(generator=lambda: map(tuple, pythonGenerator),
                                                 output_types=(tf.float32,tf.float32),
                                                 output_shapes=(tf.TensorShape((None,)), tf.TensorShape((outputs,)))
                                                )
  tfGenerator = tfGenerator.batch(batchSize)
  tfGenerator = tfGenerator.repeat(batchSize)
  tfGenerator = tfGenerator.prefetch(tf.contrib.data.AUTOTUNE)
  return tfGenerator
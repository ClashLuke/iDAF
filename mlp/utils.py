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


def mountDrive():
  from google.colab import drive
  drive.mount('/content/gdrive')

def getDatasetFromGDrive(fileName):
  import os
  os.system(''.join(['cp "/content/gdrive/My Drive/',fileName,'"']))

def getPreviousWeightsFromGDrive(weightFolderName):
  import os
  os.system(''.join(['cp -r "/content/gdrive/My Drive/',weightFolderName,'" .']))


import argparse
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp
import nltk

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="none", help="no help")
parser.add_argument("--ntsamples", type=int, default=1000000, help="number of training samples")
args = parser.parse_args()

INPUT_PREFIX = "./"
TRAIN_PATH = INPUT_PREFIX + "ds_training.csv"
TEST_PATH = INPUT_PREFIX + "ds_test.csv"
VAL_PATH = INPUT_PREFIX + "ds_eval.csv"
OUTPUT_PATH = "./models/models-{0}-".format(args.model)

OUTPUT_PATH = OUTPUT_PATH
TEST_PREDICTIONS_FILE = OUTPUT_PATH+"test-predictions"
EVAL_PREDICTIONS_FILE = OUTPUT_PATH+"eval-predictions"
WORD_MAX_LEN = 10000000 # We do not need this for generating LMs (see max_seq_length/slide window/stride)
NUM_CLASSES = 2

print (OUTPUT_PATH)

model_type == "mobilebert":
model_name = "google/mobilebert-uncased"

def getLabel(x):
  if (x[0] == "ham"):
    label = 0
  else:
    label = 1
  return label

def getDataset(step, ntsamples=10000000):
    words = thisset = set()
    filename = TEST_PATH
    df = pd.DataFrame(columns=["text", "labels"])
    if step == "train":
      filename = TRAIN_PATH
    elif step == "validation":
      filename = VAL_PATH
    elif step == "test":
      filename = TEST_PATH
    datafile = open(filename,"r")
    lineCounter, longestSample, trimmed = 0, 0, 0
    for line in datafile:
      x = line.split()
      label = getLabel(x)
      add = False
      startPoint, sampleLength = 0, 0
      content = ""
      for i,token in enumerate(x):
         if (token == "#T#" or token =="#B#" or token == "#A#" or token == "#D#"):
            add = False
            startPoint = i
            if (token == "#T#" and considerTitles):
               add = True
            if (token == "#B#" and considerBodies):
               add = True
            if (token == "#A#" and considerAnswers):
               add = True
            if (token == "#D#" and considerDescriptions):
               add = True
         elif (add and (i - startPoint) % 4 == 1 and i < len(x)-2 and x[i].lower() not in stop_words and x[i].isalnum()):
           sampleLength = sampleLength + 1
           if (sampleLength < WORD_MAX_LEN):
             if (modeLemmataTrueCase):
                tokenTmp = x[i+2]
                if (token == x[i+1] and token == x[i+2]):
                  tokenTmp = token
                else:
                  if (token != x[i+1]):
                     tokenTmp = x[i+1]
                if (modeSentimentAnalysis):
                  content = content + " " + tokenTmp+x[i+3]
                else:
                  content = content + " " + tokenTmp
             elif (modeRaw):
                if (modeSentimentAnalysis):
                  content = content + " " + x[i]+x[i+3]
                else:
                  content = content + " " + x[i]
             elif (modeLemmata):
                if (modeSentimentAnalysis):
                  content = content + " " + x[i+1]+x[i+3]
                else:
                  content = content + " " + x[i+1]
             elif (modeTrueCase):
                if (modeSentimentAnalysis):
                  content = content + " " + x[i+2]+x[i+3]
                else:
                  content = content + " " + x[i+2]
           else:
              trimmed = trimmed +1
              break
      if (sampleLength > longestSample):
         longestSample = sampleLength
      df.loc[lineCounter] = [content, label]
      x_ = content.split()
      for token in x_:
         words.add(token)
      if step == "train" and lineCounter > ntsamples:
         break
      #print ("%i %s", (lineCounter,df.loc[lineCounter]))
      lineCounter = lineCounter + 1
    datafile.close()

    DICT_SIZE = len(words)
    
    print ('Tokens in longest ' + str(step) + ' sample: ' + str(longestSample))
    print ('Samples in ' + str(step) + ': ' + str(lineCounter))
    print ('Samples timmed ' + str(trimmed) + ' at ' + str(WORD_MAX_LEN))
    print ('Number of words ' + str(DICT_SIZE))

    return df, DICT_SIZE

def saveDataset(df, filename):
   f = open(filename,"w+")
   for x in df.itertuples():
      f.write(str(x[1])+'\n')
   f.close()


def compute_metrics(allPredictions, allLabels):
    sum_rr, sum_accuracy, sampleCounter = 0, 0, 0

    fileOutput = ""

    realPerClass     = np.zeros(NUM_CLASSES)
    assignedPerClass = np.zeros(NUM_CLASSES)
    correctPerClass  = np.zeros(NUM_CLASSES)

    for prediction in allPredictions:
       rankingPosition, aux, best = 0, 0, allLabels[sampleCounter]
       realPerClass[best] = realPerClass[best] + 1
       for score in prediction:
          if (score >= prediction[allLabels[sampleCounter]]):
             rankingPosition = rankingPosition + 1
             if (score > prediction[best]):
                best = aux
          aux = aux + 1
       if (rankingPosition == 1):
         sum_accuracy = sum_accuracy + 1
         correctPerClass[best] =  correctPerClass[best] + 1
       sum_rr = sum_rr + 1.00/rankingPosition
       assignedPerClass[best] = assignedPerClass[best] + 1
       fileOutput = fileOutput + str(sampleCounter)+' '+str(allLabels[sampleCounter])+' '+str(rankingPosition)+' '+str(best)+' '+str(prediction)+'\n'
       sampleCounter = sampleCounter + 1
    macroF1Score, beta, mcc_p, mcc_t, mcc_tp = 0.00, 1.00, 0.00, 0.00, 0.00
    for counter in range(NUM_CLASSES):
       precision, recall = 0, 0
       if (realPerClass[counter] > 0):
         recall = 1.00* correctPerClass[counter]/realPerClass[counter]
       if (assignedPerClass[counter]>0):
         precision = 1.00* correctPerClass[counter]/assignedPerClass[counter]
       if (recall!=0 or precision!=0):
         macroF1Score = macroF1Score + (1+beta*beta)*recall*precision/(recall+beta*beta*precision)
       mcc_p  = mcc_p  + (realPerClass[counter] * realPerClass[counter])
       mcc_t  = mcc_t  + (assignedPerClass[counter] * assignedPerClass[counter])
       mcc_tp = mcc_tp + (assignedPerClass[counter] * realPerClass[counter])
    macroF1Score = macroF1Score / NUM_CLASSES
    test_accuracy = sum_accuracy / sampleCounter
    test_mrr = sum_rr / sampleCounter
    
    mcc =  (sampleCounter ** 2 - mcc_p) * (sampleCounter ** 2 - mcc_t)
    if mcc != 0:
      mcc =  (sum_accuracy*sampleCounter - mcc_tp) / np.sqrt(mcc)
     
    #all this code is computing aucs //binary and multiclass
    auc_score = 0.0
    if (NUM_CLASSES==2):
       allPositiveScores = []
       max_score, min_score, pos_label = allPredictions[0][0], allPredictions[0][0], 0

       for prediction in allPredictions:
          for i in range(NUM_CLASSES):
            if (max_score < prediction[i]):
               max_score = prediction[i]
            if (min_score > prediction[i]):
               min_score = prediction[i]
               
       max_score = max_score + abs(min_score)
       min_score = min_score + abs(min_score)

       for prediction in allPredictions:
          allPositiveScores.append((prediction[pos_label]+abs(min_score))/(max_score-min_score))
          
       fpr, tpr, _ = roc_curve(allLabels, allPositiveScores, pos_label=pos_label)
       auc_score = auc(fpr, tpr)
    else: #Multiclass-case AUCs
       allScores = [[0 for x in range(NUM_CLASSES)] for y in range(sampleCounter)]
       max_score, min_score = allPredictions[0][0], allPredictions[0][0]

       for s in range(0,sampleCounter):
          prediction = allPredictions[s]
          for i in range(NUM_CLASSES):
            if (max_score < prediction[i]):
               max_score = prediction[i]
            if (min_score > prediction[i]):
               min_score = prediction[i]
            allScores[s][i] = prediction[i]

       max_score = max_score + abs(min_score)
       min_score = min_score + abs(min_score)
       
       for s in range(0,sampleCounter):
          sumScore = 0.0
          for i in range(NUM_CLASSES):
            allScores[s][i] = (allScores[s][i]+abs(min_score))/(max_score-min_score)
            sumScore = sumScore + allScores[s][i]
          for i in range(NUM_CLASSES):
            if (sumScore==0):
               allScores[s][i] = 1.00 / NUM_CLASSES
            else:
               allScores[s][i] = allScores[s][i] / sumScore
            
       auc_score = roc_auc_score(allLabels, allScores, multi_class='ovr')
       
    allResults = '=> MRR:' + str(test_mrr) + ' Accuracy:' + str(test_accuracy) + ' Macro F1 Score:' + str(macroF1Score) + ' AUC:' + str(auc_score) + ' MCC:' + str(mcc)
    return test_accuracy, fileOutput, allResults


 
def generate_graphs(allPredictions, allLabels, rocFilenamePrefix, classesNames=None, colorsNames=None):
  if (NUM_CLASSES==2):
    allPositiveScores = []
    max_score, min_score, pos_label = allPredictions[0][0], allPredictions[0][0], 0
       
    for prediction in allPredictions:
      for i in range(NUM_CLASSES):
        if (max_score < prediction[i]):
          max_score = prediction[i]
        if (min_score > prediction[i]):
          min_score = prediction[i]

    max_score = max_score + abs(min_score)
    min_score = min_score + abs(min_score)

    for prediction in allPredictions:
      allPositiveScores.append((prediction[pos_label]+abs(min_score))/(max_score-min_score))

    fpr, tpr, _ = roc_curve(allLabels, allPositiveScores, pos_label=pos_label)
    auc_score = auc(fpr, tpr)

    plt.figure()
    graphLabel = 'AUC={0:.3f}'.format(auc_score) 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=graphLabel)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic ')
    plt.legend(loc="lower right")
    plt.savefig(rocFilenamePrefix+"-roc.png")

    precision, recall, _ = precision_recall_curve(allLabels, allPositiveScores, pos_label=pos_label)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.savefig(rocFilenamePrefix+"-prc.png")
  else: #Multiclass-case AUCs                                                                                                                                                                             
    roc_auc, fpr, tpr = dict(), dict(), dict()

    for i in range(NUM_CLASSES):
      max_score, min_score = allPredictions[0][i], allPredictions[0][i]
      for prediction in allPredictions:
        if (max_score < prediction[i]):
          max_score = prediction[i]
        if (min_score > prediction[i]):
          min_score = prediction[i]
          
      max_score = max_score + abs(min_score)
      min_score = min_score + abs(min_score)
       
      allPositiveScores = []
      for prediction in allPredictions:
         allPositiveScores.append((prediction[i]+abs(min_score))/(max_score-min_score))
      fpr[i], tpr[i], _ = roc_curve(allLabels, allPositiveScores, pos_label=i)
      roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
      mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],label='macro-average (AUC={0:0.3f})' ''.format(roc_auc["macro"]),
      color='navy', linestyle=':', linewidth=4)

    if colorsNames is None:
       colorsNames = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink']
       #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink']) from itertools import cycle
    for i, color in zip(range(NUM_CLASSES), colorsNames):
      if classesNames is None:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
          label='Class {0} (AUC={1:0.3f})' ''.format(i, roc_auc[i]))
      else:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
          label='Class {0} (AUC={1:0.3f})' ''.format(classesNames[i], roc_auc[i]))
      
      plt.plot([0, 1], [0, 1], 'k--', lw=2)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.0])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic')
      plt.legend(loc="lower right")
      plt.savefig(rocFilenamePrefix+"-roc.png")

if __name__ == "__main__":
   print(str(getDataset('train')))
   print(str(getDataset('validation')))


#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#examples-using-sklearn-metrics-roc-curve
#https://docs.w3cub.com/scikit_learn/auto_examples/model_selection/plot_roc.html

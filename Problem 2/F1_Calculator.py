import numpy as np


#implement F1 score calculator: https://en.wikipedia.org/wiki/F-score

def calculate_f1(values, labels, threshold):
    predic = [1 if i >=threshold else 0 for i in values]  
    TP = np.sum(np.multiply([i==True for i in labels], predic))
    TN = np.sum(np.multiply([i==False for i in predic], [not(j) for j in labels]))
    FP = np.sum(np.multiply([i==True for i in predic], [not(j) for j in labels]))
    FN = np.sum(np.multiply([i==False for i in predic], labels))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    if precision != 0 and recall != 0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0


######### TEST ##########
values = np.array([2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0])
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
threshold = 1  # Any item in `vales` above 1 is predicted to be the positive class.
print(calculate_f1(values,labels,threshold))
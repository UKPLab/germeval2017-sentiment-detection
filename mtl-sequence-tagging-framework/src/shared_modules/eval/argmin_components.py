"""
    Evaluation script for argumentation mining components.
    Based on code by SE.
"""
import logging

import numpy as np


# CHANGE: readDocsFine now expects an array of lines instead of a filename
# CHANGE: added delimiter option
def readDocsFine(lines, field, delimiter="\t"):
    docs = []
    doc = [[]]
    argTypes = []
    atype = []
    lastLabel = None
    for line in lines:
        line = line.strip()
        if line == "":
            if doc != [[]]:
                docs.append(doc)
                argTypes.append(atype)
            doc = [[]]
            atype = []
            lastLabel = None
        else:
            x = line.split(delimiter)
            label = x[field]
            if label.startswith("B-"):  # and lastLabel!="O" and lastLabel:
                atype.append(label.split(":")[0])
                if doc[-1] != []:
                    doc.append([])
            elif label.startswith("O") and lastLabel != "O" and lastLabel:
                atype.append(None)
                if doc[-1] != []:
                    doc.append([])
            elif label.startswith("O") and lastLabel != "O":
                atype.append(None)
            doc[-1].append(line)
            lastLabel = label[0]
    if doc != [[]]:
        docs.append(doc)
        argTypes.append(atype)
    return docs, argTypes


def extractComponents(lst, types):
    h = {}
    index = 0

    for ic, c in enumerate(lst):
        if types[ic] != None:
            h[index] = (types[ic], len(c))
        index += len(c)
    return h


# compute TP, FP, and FN
# as described in "End-to-End Argumentation Mining in Student Essays"
def getTP(pred, truth):
    TP = 0
    FP = 0
    FN = 0
    for x in pred:
        if x in truth:
            if truth[x] == pred[x]:

                TP += 1
            else:
                FP += 1
        else:
            FP += 1
    for x in truth:
        if x not in pred:
            FN += 1
        else:
            if truth[x] != pred[x]:
                FN += 1
    return TP, FP, FN


def checkApproxMatch(a, b, ratio=0.5):
    start_pos_a = a[0]
    start_pos_b = b[0]
    type_a = a[1][0]
    type_b = b[1][0]
    len_a = a[1][1]
    len_b = b[1][1]

    if type_a != type_b: return False
    a_tok = set(range(start_pos_a, start_pos_a + len_a))
    b_tok = set(range(start_pos_b, start_pos_b + len_b))
    n = len(a_tok.intersection(b_tok)) * 1.0

    if n / max(len(a_tok), len(b_tok)) > ratio: return True
    return False


def getTP_approx(pred, truth):
    TP = 0
    FP = 0
    FN = 0
    for x in pred:
        if x in truth:
            if truth[x] == pred[x]:
                print( x, truth[x])
                TP += 1
            else:
                found = False
                for y in truth:
                    if checkApproxMatch((x, pred[x]), (y, truth[y])):
                        TP += 1
                        found = True
                        break
                if found == True: continue
                FP += 1
        else:
            found = False
            for y in truth:
                if checkApproxMatch((x, pred[x]), (y, truth[y])):
                    TP += 1
                    found = True
                    break
            if found == True: continue
            FP += 1
    for x in truth:
        found = False
        for y in pred:
            if checkApproxMatch((x, truth[x]), (y, pred[y])):
                found = True
                break
        if found == True: continue
        FN += 1
    return TP, FP, FN


def getTP_approx_simple(pred, truth, ratio=0.5):
    TP = 0
    FP = 0
    FN = 0
    for x in pred:
        found = False
        for y in truth:
            if checkApproxMatch((x, pred[x]), (y, truth[y]), ratio=ratio):
                found = True
                if pred[x] != truth.get(x, None):
                    pass
                break
        if found == False: FP += 1
    for x in truth:
        found = False
        for y in pred:
            if checkApproxMatch((x, truth[x]), (y, pred[y]), ratio=ratio):
                found = True
                break
        if found == True:
            TP += 1
        else:
            FN += 1
    return TP, FP, FN


# CHANGE: instead of calling this file as a script, the evaluation is triggered by a method call
def evaluate_argmin_components(
        prediction_lines,
        truth_lines,
        prediction_column,
        truth_column,
        ratio=0.5,
        delimiter="\t"
):
    """
    Evaluate the prediction results w.r.t. AM components.

    Args:
        prediction_lines (`list` of str): a list of lines as they would occur in a CoNLL formatted file.
            The labels are predictions.
        truth_lines (`list` of str): a list of lines as they would occur in a CoNLL formatted file.
            The labels are gold standard.
        prediction_column (int): The column of the prediction lines in which the predicted label can be found.
        truth_column (int): The column of the truth lines in which the predicted label can be found.
        ratio (float, optional): The ratio. Defaults to 0.5
        delimiter (str, optional): The delimiter for splitting a line into its components. Defaults to tab.

    Returns:
        `tuple` of float: The calculated scores: true positives, false positives, false negatives, f1, mean of lengths
    """
    # CHANGE: added logger
    logger = logging.getLogger("shared.argmin_components.evaluate_argmin_components")

    # CHANGE: removed reading file names from command line
    predDocs, argTypesDocs = readDocsFine(prediction_lines, prediction_column, delimiter=delimiter)
    truthDocs, argTypesDocsTruth = readDocsFine(truth_lines, truth_column, delimiter=delimiter)

    logger.debug("Prediction docs: %d", len(predDocs))
    logger.debug("Truth docs: %d", len(truthDocs))

    TPS = 0
    FPS = 0
    FNS = 0
    printLocalF1 = True
    lengths = []

    # CHANGE: removed reading ratio from command line

    for idoc, doc in enumerate(predDocs):

        if len(doc) != len(argTypesDocs[idoc]):
            logger.debug("PROBLEM: %s", idoc)
        try:
            pred_c = extractComponents(doc, argTypesDocs[idoc])
            truth_c = extractComponents(truthDocs[idoc], argTypesDocsTruth[idoc])

        except IndexError:
            logger.error("ERROR in doc %d", idoc)
            continue
        for q in truth_c:
            lengths.append(truth_c[q][-1])
        TP, FP, FN = getTP_approx_simple(pred_c, truth_c, ratio=ratio)
        TPS += TP
        FPS += FP
        FNS += FN
        if printLocalF1:
            denom = max(2 * TP + FP + FN, 1)
            # logger.debug("::: %f", 2 * TP * 1.0 / denom)
    F1 = 2 * TPS * 1.0 / (2 * TPS + FPS + FNS)
    return TPS, FPS, FNS, F1, np.mean(lengths)

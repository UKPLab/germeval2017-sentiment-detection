"""
    Evaluation script for argumentation mining relations.
    Based on code by SE.
"""
import logging

import numpy as np


# CHANGE: readDocsFine now expects an array of lines instead of a filename
# CHANGE: added delimiter option
import sys


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


def findLastMajorClaim(lst, types):
    index = 0
    c = 0
    for ix, x in enumerate(types):
        if x == "B-MajorClaim":
            index = c
        c += len(lst[ix])
    return index


# CHANGE: added column parameter
def extractRelations(lst, types, column):
    h = {}
    lmc = findLastMajorClaim(lst, types)
    index = 0
    for ic, c in enumerate(lst):
        if types[ic] == "B-MajorClaim":
            h[index] = (None, 0)
        elif types[ic] == "B-Claim":
            stance = lst[ic][0].split("\t")[column].split(":")[-1]
            h[index] = ("B-Claim:" + stance, lmc)
        elif types[ic] == "B-Premise":
            stance = lst[ic][0].split("\t")[column].split(":")[-1]

            rel = lst[ic][0].split("\t")[column].split(":")[1]
            h[index] = ("B-Premise:" + stance, int(rel) - 1)
        index += len(c)
    return h


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


def checkApproxMatch(a, b, ratio=0.5, typeRequirement=True):
    start_pos_a = a[0]
    start_pos_b = b[0]
    type_a = a[1][0]
    type_b = b[1][0]
    len_a = a[1][1]
    len_b = b[1][1]

    # root components
    if len_a is None or len_b is None:
        return len_a == len_b

    if type_a != type_b and typeRequirement: return False
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


def computeF1_relations(pred, truth):
    TP = 0;
    FP = 0;
    FN = 0;
    for x in truth:
        found = False
        for y in pred:
            if x == y and truth[x] == pred[y]:
                TP += 1
                found = True
                break
        if not found: FN += 1
    for y in pred:
        found = False
        for x in truth:
            if x == y and truth[x] == pred[y]:
                found = True
                break
        if not found: FP += 1
    return TP, FP, FN


def computeF1_relations_approx(pred, truth, pred_c, truth_c, ratio=0.5):
    TP = 0;
    FP = 0;
    FN = 0;

    for x in truth:
        found = False
        for y in pred:
            source_x = truth_c[x]
            t_x = truth[x][-1]
            if t_x != 0:  # 0 is ROOT
                try:
                    target_x = truth_c[t_x]
                except KeyError:
                    print (t_x, truth_c, "EXIT")
                    sys.exit(1)
            else:
                target_x = (None, None)
            source_y = pred_c[y]
            t_y = pred[y][-1]
            if t_y != 0:  # 0 is ROOT
                target_y = pred_c.get(t_y, None)
                if target_y == None:  # links to non-argumentative component
                    continue
            else:
                target_y = (None, None)
            # relation must match
            # and source components must approximately match
            # and target components must approximately match
            try:
                rel_x = truth[x][0]  # .split(":")[0]
            except AttributeError:
                rel_x = None
            try:
                rel_y = pred[y][0]  # .split(":")[0]
            except AttributeError:
                rel_y = None
            if rel_x == rel_y and checkApproxMatch((x, source_x), (y, source_y), ratio=ratio,
                                                   typeRequirement=False) and checkApproxMatch((t_x, target_x),
                                                                                               (t_y, target_y),
                                                                                               ratio=ratio,
                                                                                               typeRequirement=False):
                if source_x[0] != source_y[0] or True:

                    pass
                TP += 1
                found = True
                break
        if not found: FN += 1
    for y in pred:
        found = False
        source_y = pred_c[y]
        t_y = pred[y][-1]
        if t_y != 0:  # 0 is ROOT
            target_y = pred_c.get(t_y, None)
            if target_y is None:

                FP += 1
                continue
        else:
            target_y = (None, None)
        for x in truth:
            source_x = truth_c[x]
            t_x = truth[x][-1]
            if t_x != 0:  # 0 is ROOT
                target_x = truth_c[t_x]
            else:
                target_x = (None, None)
            source_y = pred_c[y]
            # relation must match
            # and source components must approximately match
            # and target components must approximately match
            try:
                rel_x = truth[x][0]  # .split(":")[0]
            except AttributeError:
                rel_x = None
            try:
                rel_y = pred[y][0]  # .split(":")[0]
            except AttributeError:
                rel_y = None
            if rel_x == rel_y and checkApproxMatch((x, source_x), (y, source_y), ratio=ratio,
                                                   typeRequirement=False) and checkApproxMatch((t_x, target_x),
                                                                                               (t_y, target_y),
                                                                                               ratio=ratio,
                                                                                               typeRequirement=False):
                if source_x[0] != source_y[0]:

                    pass
                found = True
                break
        if not found: FP += 1
    return TP, FP, FN


# CHANGE: instead of calling this file as a script, the evaluation is triggered by a method call
def evaluate_argmin_relations(
        prediction_lines,
        truth_lines,
        prediction_column,
        truth_column,
        ratio=0.5,
        delimiter="\t"
):
    """
    Evaluate the prediction results w.r.t. AM relations.

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

    TPS = 0
    FPS = 0
    FNS = 0
    lengths = []

    # CHANGE: removed reading ratio from command line

    for idoc, doc in enumerate(predDocs):

        if len(doc) != len(argTypesDocs[idoc]):
            logger.debug("PROBLEM: %s", idoc)
        try:
            pred_c = extractComponents(doc, argTypesDocs[idoc])
            # CHANGE: provide column
            pred_rel = extractRelations(doc, argTypesDocs[idoc], prediction_column)
        except IndexError:
            logger.error("Something wrong with doc %d", idoc)
            continue
        truth_c = extractComponents(truthDocs[idoc], argTypesDocsTruth[idoc])
        # CHANGE: provide column
        truth_rel = extractRelations(truthDocs[idoc], argTypesDocsTruth[idoc], truth_column)
        for q in truth_c:
            lengths.append(truth_c[q][-1])

        TP, FP, FN = computeF1_relations_approx(pred_rel, truth_rel, pred_c, truth_c, ratio=ratio)

        TPS += TP
        FPS += FP
        FNS += FN
    F1 = 2 * TPS * 1.0 / (2 * TPS + FPS + FNS)
    return TPS, FPS, FNS, F1, np.mean(lengths)

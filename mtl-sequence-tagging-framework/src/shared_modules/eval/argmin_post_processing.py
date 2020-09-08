"""
    Module for postprocessing of argumentation mining predictions.
    Based on code by SE.
"""

import sys

from .argmin_fix_inconsistencies import fix_docs


# SAMPLE USAGE:
# ./relative2absolute.py ../truth/relative/truthessay005.xmitypes.conll.xmi.conll

# CHANGE: added parameter label_column
def readComponents(lst, label_column):
    arguments = []
    index = 0
    rel2abs = {}
    for line in lst:
        line = line.strip()
        line_components = line.split()
        label = line_components[label_column]
        if label.startswith("B-"):
            tt = label.split(":")
            relation = None
            if len(tt) > 1:
                try:
                    relation = int(tt[1])
                except ValueError:
                    relation = None
            arguments.append((index, relation))
            # rel2abs[index] = relation
        index += 1
    for (q, x) in enumerate(arguments):
        i, rel = x
        if rel != None:
            absolute = arguments[q + rel][0]
        else:
            absolute = None
        rel2abs[i] = absolute
    return arguments, rel2abs


# CHANGE: added parameters word_column and label_column
def rewriteAbs(lst, rel2abs, word_column, label_column):
    i = 0
    output_lines = []

    for line in lst:
        line = line.strip()
        j = i + 1
        line_components = line.split()
        label = line_components[label_column]
        token = line_components[word_column]


        if label.startswith("B-"):
            absposition = rel2abs[i]
        elif label.startswith("O"):
            absposition = None
        if absposition != None:
            q = label.split(":")
            q[1] = str(absposition + 1)
            label = ":".join(q)


        # CHANGE: new line is not printed but added to the output_lines
        output_lines.append("\t".join([str(j), token, label]))
        i += 1

    # CHANGE: added return value
    return output_lines


# CHANGE: readDocs expects a list of lines or an opened file instead of a filename
def readDocs(lines):
    docs = []
    doc = []
    for line in lines:
        if line.strip() == "":
            if doc != []: docs.append(doc)
            doc = []
        else:
            doc.append(line)
    if doc != []: docs.append(doc)
    return docs


# CHANGE: added method for programmatic use
def relative_2_absolute(lines, word_column, label_column):
    """
    Convert relative indices to absolute indices in AM documents.
    The output format is:
        [index]\t[word]\t[label]

    Documents are separated by empty lines (empty => not evan an index)

    Args:
        lines (`list` of str): Lines that contain words and labels (and potentially more information),
        word_column (int): Column index for words
        label_column (int): Column index for labels

    Returns:
        `list` of str: lines in the output format
    """
    docs = readDocs(lines)
    docs = fix_docs(docs, word_column, label_column)
    # At this point, every line within the document is as follows:
    #   index   token   label
    # => word column is 1 and label column is 2
    updated_lines = []

    for doc in docs:
        args = readComponents(doc, 2)
        rel2abs = args[1]

        # Add updated lines for document
        updated_lines.extend(rewriteAbs(doc, rel2abs, 1, 2))

        # Add an empty line after the document
        updated_lines.append("")

    return updated_lines


"""
This module contains methods to fix inconsistencies within AM predictions.
Inconsistencies include:
 * Wrong BIO tag format
 * Inconsistent reference numbers
 * Out of document references

The module implements the corrections as described in the appendix of https://arxiv.org/pdf/1704.06104.pdf
"""

import logging
import os

import sys


class Line(object):
    """
    Objects of this class represent a line with an index, a token, and a label.
    """

    def __init__(self, idx, token, label):
        """
        Initialize a line.
        Args:
            idx (int): Line index
            token (str): Token
            label (Label): Label
        """
        assert isinstance(idx, int)
        assert isinstance(token, str) or isinstance(token, unicode), \
            "Expected token to be a string or unicode, but got %s (%s)" % (token, type(token))
        assert isinstance(label, Label)

        self.idx = idx
        self.token = token
        self.label = label

    def __str__(self):
        return "\t".join([str(self.idx), self.token, self.label.__str__()])


class Label(object):
    """
    Objects of this class represent an AM label. All components of the label can be modified individually.
    The `__str__` method allows to convert the object to a valid string representation that can be used in prediction
    files.

    The original label is always retrained within the `orig_label_str` property.
    """
    def __init__(self, orig_label_str):
        """
        Initialize the label
        Args:
            orig_label_str (str): Original label
        """
        self.orig_label_str = orig_label_str
        self.is_o = orig_label_str == "O"
        self.is_i = orig_label_str.startswith("I-")
        self.is_b = orig_label_str.startswith("B-")

        self.prefix = "B-" if self.is_b else "I-" if self.is_i else "O"

        self.type = None
        self.ref = None
        self.rel_type = None

        if self.is_i or self.is_b:
            components = orig_label_str.split(":")

            self.type = components[0].split("-")[1]

            if self.type == "Premise":
                self.rel_type = components[2]
                self.ref = components[1]
            elif self.type == "Claim" and len(components) > 1:
                self.rel_type = components[1]

    def __str__(self):
        if self.is_o:
            return "O"

        if self.ref is None and self.rel_type is None:
            # A label without a reference value or a relation type
            # Usually for "MajorClaim"
            return self.prefix + self.type

        if self.ref is None:
            # A label without a reference value
            # Usually for "Claim"
            return self.prefix + self.type + ":" + self.rel_type

        # A label with reverence value and relation type
        return self.prefix + self.type + ":" + str(self.ref) + ":" + self.rel_type


def has_bio_error(prev_label, curr_label):
    """
    Check whether a BIO error occurred by comparing the current label with the previous label.
    Examples for incorrect BIO tags:
        1. I- follows I- of another type
           1    Hello   B-ABC
           2    World   I-ABC
           3    !       I-XYZ
        2. I- after O
           1    Hello   O
           2    World   I-ABC
           3    !       I-ABC

    Args:
        prev_label (Label): Previous label
        curr_label (Label): Current label

    Returns:
        bool: Whether or not a BIO error occurred
    """
    i_follows_o = prev_label.is_o and curr_label.is_i
    i_follows_other_component = all([
        curr_label.is_i,
        any([
            # The types differ
            prev_label.type != curr_label.type,
            # OR: the types are equal, but the relation type differs
            all([
                prev_label.type == curr_label.type,
                prev_label.rel_type is not None,
                curr_label.rel_type is not None,
                prev_label.rel_type != curr_label.rel_type
            ])
        ])
    ])

    return i_follows_o or i_follows_other_component


def fix_component_consistency(component):
    """
    Fix inconsistent values for the reference property within a component, i.e. if there are different
    reference values within a single component, choose the majority value and apply it to all lines.

    NOTE: this method mutates the lines in `component`.

    Args:
        component (`list` of Line): A list of lines that constitute the component.

    Returns:
        (`list` of Line, bool): The updated component and a flag that indicates whether or not something was fixed
    """
    if len(component) == 0:
        # The component has no lines --> return early
        return component, False

    first = component[0]
    if first.label.ref is None:
        # The component does not have reference values (e.g. a Claim label) --> return early
        return component, False

    refs = [line.label.ref for line in component]

    if all(ref == first.label.ref for ref in refs):
        # The reference values are already consistent --> return early
        return component, False

    # Count the different reference values to find the most frequent one
    ref_counts = {}

    for ref in refs:
        if ref not in ref_counts:
            ref_counts[ref] = 1
        else:
            ref_counts[ref] += 1

    # Find the most frequent reference value
    majority_ref = 0
    majority_ref_count = -1

    for ref, count in ref_counts.items():
        if count >= majority_ref_count:
            majority_ref = ref
            majority_ref_count = count

    # Update all lines with the new majority value
    return update_component_ref(component, majority_ref), True


def update_component_ref(component, ref):
    """
    Update the reference property for each line in a component.
    NOTE: this method mutates the lines in `component`.

    Args:
        component (`list` of Line): A list of lines
        ref (int or str): The new value for the reference property

    Returns:
        `list` of Line: the updated component.
    """
    for line in component:
        line.label.ref = ref

    return component


def fix_single_doc(doc, word_column, label_column):
    """
    Fix inconsistencies in the provided document.
    Returns a list of fixed lines where each line is as follows:
        Index   Token   Label

    Args:
        doc (`list` of str): A list of lines
        word_column (int): Which column contains words
        label_column (int): Which column contains labels

    Returns:
        (`list` of str, int, int, int): A tuple containing a list of fixed lines, number of BIO errors, number of
            consistency errors, and number of out of document errors.
    """
    # logger = logging.getLogger("shared.argmin_fix_inconsistencies.fix_single_doc")
    # logger.debug("Fixing document with %d lines", len(doc))

    # Metrics
    bio_errors = 0
    consistency_errors = 0
    out_of_document_errors = 0

    # Helper variables
    prev_label = Label("O")
    # The list of lines represents the fixed document
    # It will be filled incrementally while iterating over the original document and fixing errors
    lines = []
    arguments = []
    component = []

    for idx, line in enumerate(doc):
        line = line.strip()

        assert line != "", "Expected line not to be empty."

        line_components = line.split("\t")
        token = line_components[word_column]
        label = line_components[label_column]

        # Convert to a label object
        label = Label(label)

        # Check for BIO errors and fix them if necessary
        if has_bio_error(prev_label, label):
            bio_errors += 1
            label.prefix = "B-"

        if label.prefix == "B-" or (label.prefix == "O" and prev_label.prefix != "O"):
            # A component finished before
            component, consistency_fixed = fix_component_consistency(component)

            if consistency_fixed:
                consistency_errors += 1

            # Add lines of component to list of lines
            lines.extend(component)

            if prev_label.prefix != "O":
                # Previous component was not a list of O
                arguments.append(component)

            # Start new component with current line
            component = [Line(idx, token, label)]
        else:
            # Add line to current component
            component.append(Line(idx, token, label))

        # Create the previous label as a new object
        # from the original (i.e. unmodified) label
        # string.
        prev_label = Label(label.orig_label_str)

    if component != []:
        # Add last component
        lines.extend(component)
        if component[0].label.prefix == "B-":
            arguments.append(component)

    # Iterate over all arguments in the document and try to find out of document references
    # Updates are applied by reference --> mutation (be careful!)
    for idx, arg in enumerate(arguments):
        ref = arg[0].label.ref

        if ref is None:
            continue

        ref = int(ref)
        abs_ref_idx = idx + ref

        if abs_ref_idx < 0:
            # Reference points to an index before the document
            out_of_document_errors += 1
            update_component_ref(arg, ref - abs_ref_idx)

        if abs_ref_idx >= len(arguments):
            # Reference points to an index after the document
            out_of_document_errors += 1
            update_component_ref(arg, ref - (abs_ref_idx - (len(arguments) - 1)))

    # logger.debug("Document has %d arguments.", len(arguments))

    return [line.__str__() for line in lines], bio_errors, consistency_errors, out_of_document_errors


def fix_docs(docs, word_column, label_column):
    """
    Fix inconsistencies in the provided documents.
    Returns a list of fixed documents where each line is as follows:
        Index   Token   Label

    Tab is used as the separator

    Args:
        docs (`list` of `list` of str): A list of documents
        word_column (int): Which column contains words
        label_column (int): Which column contains labels

    Returns:
        `list` of `list` of str: A list of fixed documents
    """
    num_docs = len(docs)

    logger = logging.getLogger("shared.argmin_fix_inconsistencies.fix_docs")
    logger.debug("Fixing %d documents with %d lines in total", num_docs, sum([len(doc) for doc in docs]))

    bio_errors = 0
    consistency_errors = 0
    out_of_document_errors = 0

    for i in xrange(num_docs):
        fixed_doc, be, ce, oode = fix_single_doc(docs[i], word_column=word_column, label_column=label_column)
        docs[i] = fixed_doc
        bio_errors += be
        consistency_errors += ce
        out_of_document_errors += oode

    logger.debug("Fixed %d BIO errors", bio_errors)
    logger.debug("Fixed %d consistency errors", consistency_errors)
    logger.debug("Fixed %d out of document references", out_of_document_errors)

    return docs


def main():
    """
    This method is used when this file is called directly as a script.
    It expects command line parameters:
        * Index of word column
        * Index of label column
        * A file path

    The method will create corrected versions of the files (.corr file extension).
    """
    assert len(sys.argv) == 4
    word_column = int(sys.argv[1])
    label_column = int(sys.argv[2])
    file_path = sys.argv[3]

    docs = []
    with open(file_path, mode="r") as f:
        doc = []
        for line in f:
            line = line.strip()

            if line == "":
                if len(doc) != 0:
                    docs.append(doc)
                doc = []
            else:
                doc.append(line)

    fixed_docs = fix_docs(docs, word_column, label_column)

    with open(file_path + ".corr", mode="w") as f:
        for doc in fixed_docs:
            for line in doc:
                f.write(line)
                f.write(os.linesep)

            f.write(os.linesep)

if __name__ == "__main__":
    main()

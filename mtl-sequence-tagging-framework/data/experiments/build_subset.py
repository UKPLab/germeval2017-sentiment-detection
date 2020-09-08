"""
Module to extract a subset of documents from an input file.
"""

import sys


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python build_subset.py FRACTION < INPUT_FILE > OUTPUT_FILE\n")
        exit(-1)

    docs = []
    doc = []

    for line in sys.stdin:
        if line.strip() == "":
            if len(doc) != 0:
                docs.append(doc)
                doc = []
        else:
            doc.append(line)

    if len(doc) != 0:
        docs.append(doc)

    num_docs = len(docs)
    num_docs_subset = int(float(sys.argv[1]) * num_docs)
    sys.stderr.write("Read %d docs\n" % num_docs)
    sys.stderr.write("Writing a subset of %d docs to stdout\n" % num_docs_subset)

    for i in xrange(num_docs_subset):
        for line in docs[i]:
            sys.stdout.write(line)

        # Add empty line
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()

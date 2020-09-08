"""
This script converts an input stream (stdin) with two columns into an output stream (stdout) with two columns, but
every label is preceeded by "B-". Such files are only necessary for the architecture by Lample et al. which expects
all files to be in BIO format.

Empty lines remain unchanged. Columns will be separated by tabs.
"""
import os
import sys

if __name__ == "__main__":
    word_column = 0
    label_column = 1

    for line in sys.stdin:
        line = line.strip()

        if line == "":
            sys.stdout.write(os.linesep)
            continue

        cells = line.split()
        sys.stdout.write("%s%sB-%s%s" % (
            cells[word_column], "\t",
            cells[label_column], os.linesep
        ))

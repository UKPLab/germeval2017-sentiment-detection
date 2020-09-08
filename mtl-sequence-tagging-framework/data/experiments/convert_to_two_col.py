"""
This script converts an input stream (stdin) with multiple columns into an output stream (stdout) with two columns:
token + label.

The script requires two parameters: index of the token column and index of the label column.
Empty lines remain unchanged. Columns will be separated by tabs.
"""
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Please provide the necessary arguments: word column and label column.\n")
        exit(-1)

    word_column = int(sys.argv[1])
    label_column = int(sys.argv[2])

    for line in sys.stdin:
        line = line.strip()

        if line == "":
            sys.stdout.write(os.linesep)
            continue

        cells = line.split()
        sys.stdout.write("%s%s%s%s" % (
            cells[word_column], "\t",
            cells[label_column], os.linesep
        ))

"""Data reader for files in the CONLL format"""

import codecs
import logging
from os import path
import pickle as pkl

from .BaseDataReader import BaseDataReader
from .Sample import Sample
from ..data.preprocess import word_normalize
from ..config.FileConfig import FileConfig
from ..constants import \
    DATA_OUT_RAW, DATA_OUT_INDEX, DATA_OUT_PADDED, DATA_OUT_PADDED_RIGHT,\
    DATA_TYPE_TRAIN, DATA_TYPE_DEV, DATA_TYPE_TEST,\
    DOCSTART, TOKEN_UNKNOWN


class ConllDataReader(BaseDataReader):
    """
    Data reader implementation that can read files in CONLL format.
    """

    def __init__(self):
        self._files = None
        self._data = {DATA_OUT_PADDED_RIGHT: {}, DATA_OUT_PADDED: {}, DATA_OUT_INDEX: {}, DATA_OUT_RAW: {}}
        self._words = None
        self._labels = None
        self._word2idx = None
        self._label2idx = None
        self._max_sequence_length = None
        self._paths = {}
        self._name = ""

    def add_name(self, name):
        """
        Set a name for this data reader. This is necessary for unique pickle files.
        Args:
            name (str): A unique name (e.g. the task name)
        """
        assert isinstance(name, str)
        self._name = name

    def add_files(self, files):
        """Add files to this data reader.

        Args:
            files (`dict` of FileConfig): A dictionary of files that maps file configs to train, dev, and test.

        Returns:
            None
        """
        logger = logging.getLogger("shared.conll_data_reader.add_files")

        # Ensure that the provided data is actually a dictionary of files
        assert isinstance(files, dict)
        assert all([isinstance(f, FileConfig) for f in files.values()])

        logger.debug("Adding %d file configurations to CONLL data reader.", len(files))

        self._files = files

    def add_paths(self, paths):
        """
        Add the paths for the experiment.
        Args:
            paths (`dict` of str): Necessary paths.
        """
        self._paths = paths

    def get_data(self, data_type="train", out_format="padded", word2idx=None):
        """Get the data from the corpus.

        Get the data from the corpus. The user can choose the type and format of the data.
        The data format choice allows to get data that is ready for consumption by a Keras
        network.

        Args:
            data_type (str): The type of data. One of "train", "dev" or "test".
            out_format (str): The format of the data. One of "padded" (words represented by
                their indices and padded by the index for <MASK>), "padded-right" (as "padded",
                but padding is applied from the right and not from the left) "index" (words represented
                by their indices) or "raw" (words in their original form).
            word2idx (`dict` of int): Mapping from words to indices. Only used for "padded"
                and "index". If not supplied, using `self.get_words(out_format="word2idx")`.

        Returns:
            `list` of Sample: The data.
        """
        logger = logging.getLogger("shared.conll_data_reader.get_data")

        assert data_type in [DATA_TYPE_TRAIN, DATA_TYPE_DEV, DATA_TYPE_TEST]
        assert out_format in [DATA_OUT_RAW, DATA_OUT_INDEX]
        assert self._name != ""

        logger.debug("Get data for %s in format %s", data_type, out_format)
        # print "Cached data:"
        # print "\n  - ".join(["%s -> %s" % (f, t) for f in self._data.keys() for t in self._data[f].keys()])

        if data_type not in self._data[out_format]:
            # print "Data is not in cache. Have to read it from disk."

            file_config = self.files[data_type]
            assert isinstance(file_config, FileConfig)

            pkl_path = path.join(self._paths["pkl"], "%s_%s_word-col-%d_label-col-%d_%s_%s.pkl" % (
                self._name,
                file_config.filename,
                file_config.word_column,
                file_config.label_column,
                data_type,
                out_format
            ))

            logger.debug("Pickle path is: %s", pkl_path)

            if path.isfile(pkl_path):
                logger.debug("There is a .pkl file. Load %s data in format %s from this file.", data_type, out_format)
                with open(pkl_path, 'rb') as f:
                    self._data[out_format][data_type] = pkl.load(f)

                logger.debug("Finished loading from pickle file.")
                return self._data[out_format][data_type]

            logger.debug("Pickle file cannot be located. Loading from scratch.")

            if out_format == DATA_OUT_RAW:
                logger.debug("Loading data from file.")
                logger.debug("Filename: %s", file_config.filename)
                logger.debug("Word column: %d", file_config.word_column)
                logger.debug("Label column: %d", file_config.label_column)
                logger.debug("Column separator: %s" % "space" if file_config.column_separator == " " else "tab")
                logger.debug("Output format: %s", out_format)

                # Based on https://github.com/glample/tagger/blob/master/loader.py#L8-L29
                samples = []
                sentence = []
                label_sequence = []
                docid = ''
                for line in codecs.open(file_config.path, 'r', 'utf8'):
                    line = line.rstrip()
                    if not line:
                        # Empty line
                        if len(sentence) > 0:
                            if DOCSTART not in sentence[0]:
                                # Only add sentences that do not contain "DOCSTART"
                                samples.append(Sample(sentence, label_sequence, docid=docid))
                            sentence = []
                            label_sequence = []
                            docid=''
                    else:
                        # Non-empty line
                        # If we only have 1 token, it is the document id:
                        if len(line.split(file_config.column_separator)) < 2 and line:
                            docid = line.strip()
                            continue
                        token = line.split(file_config.column_separator)
                        assert len(token) > file_config.word_column, "Invalid: %s | %d" % (token, file_config.word_column)
                        assert len(token) > file_config.label_column, "Invalid: %s | %d" % (token, file_config.label_column)
                        sentence.append(token[file_config.word_column])
                        label_sequence.append(token[file_config.label_column])

                if len(sentence) > 0:
                    if DOCSTART not in sentence[0]:
                        # Only add sentences that do not contain "DOCSTART"
                        samples.append(Sample(sentence, label_sequence))

                logger.debug("Finished loading %s raw sentences", len(samples))

                self._data[out_format][data_type] = samples
            elif out_format == DATA_OUT_INDEX:
                raw_samples = self.get_data(
                    data_type=data_type,
                    out_format=DATA_OUT_RAW,
                    word2idx=word2idx
                )
                if not word2idx:
                    logger.warn("No word-to-index mapping provided for `get_data`."
                                " Using word-to-index mapping for this task only.")
                    word2idx = self.get_words(out_format="word2idx")

                label2idx = self.get_labels(out_format="label2idx")

                samples = []

                num_unknown_tokens = 0
                num_tokens = 0

                for sample in raw_samples:
                    token_indices = []
                    for token in sample.raw_tokens:
                        num_tokens += 1

                        if token in word2idx:
                            token_indices.append(word2idx[token])
                        elif token.lower() in word2idx:
                            token_indices.append(word2idx[token.lower()])
                        elif word_normalize(token) in word2idx:
                            token_indices.append(word2idx[word_normalize(token)])
                        else:
                            num_unknown_tokens += 1
                            token_indices.append(word2idx[TOKEN_UNKNOWN])

                    label_indices = [label2idx[label] for label in sample.raw_labels]

                    samples.append(Sample(sample.raw_tokens, sample.raw_labels, token_indices, label_indices, docid=sample.docid))

                logger.debug(
                    "Loaded indices for %d sentences with %d tokens. %d tokens (%.2f%%) were unknown.",
                    len(samples),
                    num_tokens,
                    num_unknown_tokens,
                    (num_unknown_tokens / float(num_tokens)) * 100
                )

                self._data[out_format][data_type] = samples
            else:
                raise ValueError("Unknown data format")

            # print "Data written to `self._data[%s][%s]`" % (out_format, data_type)
            logger.debug("Finished loading from scratch. Writing results to .pkl file")
            with open(pkl_path, mode="wb") as f:
                pkl.dump(self._data[out_format][data_type], f, -1)

        else:
            pass
            # print "Data is in cache. Reading from memory."

        return self._data[out_format][data_type]

    def get_words(self, out_format="list"):
        """Get all words.

        Depending on `format`, either get a list of all words (over all files) or a
        dictionary of all words in the corpus mapped to indices.

        Args:
            out_format (str): The output format. One of "list" or "word2idx".

        Returns:
            `list` of str: `dict` of int: A word list or dictionary.
        """
        if not self._words:
            raw_data = [
                self.get_data(data_type=data_type, out_format=DATA_OUT_RAW)
                for data_type
                in [DATA_TYPE_TRAIN, DATA_TYPE_DEV, DATA_TYPE_TEST]
            ]

            raw_sentences = [sample.raw_tokens for samples in raw_data for sample in samples]
            words = [word for sentence in raw_sentences for word in sentence]
            # Find all unique words and add the mask as the first word, i.e. index 0
            self._words = list(set(words))

        if out_format == "list":
            return self._words

        if not self._word2idx:
            self._word2idx = {k: v for v, k in enumerate(self._words)}

        if out_format == "word2idx":
            return self._word2idx

        raise Exception("Unknown `get_words` format: %s" % out_format)

    def get_labels(self, out_format="list"):
        """Get all labels.

        Depending on `format`, either get a list of all labels (over all files) or a
        dictionary of all labels in the corpus mapped to indices.

        Args:
            out_format (str): The output format. One of "list" or "label2idx".

        Returns:
            `list` of str: `dict` of int: A label list or dictionary.
        """
        if not self._labels:
            raw_data = [
                self.get_data(data_type=data_type, out_format=DATA_OUT_RAW)
                for data_type
                in [DATA_TYPE_TRAIN, DATA_TYPE_DEV, DATA_TYPE_TEST]
            ]

            raw_label_sequences = [
                sample.raw_labels
                for samples
                in raw_data
                for sample
                in samples
            ]
            labels = [label for label_sequence in raw_label_sequences for label in label_sequence]
            # Find all unique labels and add the mask as the first word, i.e. index 0
            self._labels = list(set(labels))

        if out_format == "list":
            return self._labels

        if not self._label2idx:
            self._label2idx = {k: v for v, k in enumerate(self._labels)}

        if out_format == "label2idx":
            return self._label2idx

        raise Exception("Unknown `get_labels` format: %s" % out_format)

    def get_max_sequence_length(self):
        """Determine the maximum sequence length.

        This method determines the maximum sequence length over all documents in the corpus.
        This could be, for instance, the length of the longest sentence in the corpus.

        Returns:
            int: Maximum sequence length.
        """
        if not self._max_sequence_length:
            raw_data = [
                self.get_data(data_type=data_type, out_format=DATA_OUT_RAW)
                for data_type
                in [DATA_TYPE_TRAIN, DATA_TYPE_DEV, DATA_TYPE_TEST]
                ]

            raw_sentences = [sample.raw_tokens for samples in raw_data for sample in samples]
            sentence_lengths = [len(sentence) for sentence in raw_sentences]

            self._max_sequence_length = max(sentence_lengths)

        return self._max_sequence_length

    @property
    def files(self):
        """
            `dict` of FileConfig: List of file configs for the data.
            This is a list even if only one file name was supplied in `add_files`.
        """
        return self._files

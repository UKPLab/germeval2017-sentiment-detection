import gensim
import sys
import os

# BEWARE: this might crash your PC...

def main():
    if len(sys.argv) < 2:
        print "Usage: python convert_to_utf8.py GoogleNews-vectors-negative300.bin"
        exit(-1)

    g_news_path = sys.argv[1]
    g_news_txt_dir = os.path.dirname(g_news_path)
    g_news_txt_file = os.path.join(g_news_txt_dir, os.path.basename(g_news_path).replace(".bin", ".txt"))

    model = gensim.models.KeyedVectors.load_word2vec_format(g_news_path, binary="True")
    model.save_word2vec_format(g_news_txt_file, binary=False)


if __name__ == "__main__":
    main()

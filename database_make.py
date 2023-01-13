import math
import os
#from test.test_decimal import directory
from lxml import html
import codecs
from bs4 import BeautifulSoup
from matplotlib.pyplot import cla
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json
from pathlib import Path
from tqdm import tqdm


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download('words')

from nltk.corpus import words as english_words


WORD_DOCS_MAP_PATH = "word_docs_map.json"
DOC_PATH_WORDS_MAP_PATH = "doc_path_words_map.json"
DOC_PATH_IMPORTANT_WORDS_MAP_PATH = "doc_path_important_words_map.json"
DOC_PATH_WORD_COUNT_MAP_PATH = "doc_path_word_count_map.json"
WORD_ID_PATH = "word_id.json"

MAX_DIR_NO = 75


class Database:
    def __init__(self):
        self.word_id_map = {}
        self.word_docs_map = {}
        # doc_path: words
        self.doc_path_words_map = {} 
        self.doc_path_important_words_map = {}
        # {path: {word: count, ...}}
        self.doc_path_word_count_map = {}
        self.total_doc_count = 0 
        self.total_word_count = 0
        self.primary_dir = ''
        self.files = []
        self.data = {}
        self.stop_words = set(stopwords.words('english'))
        self.english_valid_words = set(english_words.words())
        self.word_net_lemma = WordNetLemmatizer()
        
    def collect(self, work_dir):
        self.scan_files(work_dir)
        self.load_bookkeeping(work_dir)

        for i in tqdm(range(len(self.files))):
            path = self.files[i]
            self.extract_tokens(path)

        self.fillout_idf()
        self.extract_word_id()
        self.save()
        print("District word count {}, doc count {}".format(len(self.word_docs_map), len(self.doc_path_words_map)))

    def scan_files(self, parent):
        self.primary_dir = parent
        for i in range(0, MAX_DIR_NO):
            d = os.path.join(parent, str(i))
            for f1 in os.listdir(d): #对于每一个direc 里面有500个f1. 打开每一个f1并添加到lsst里
                self.total_doc_count += 1
                d2 = os.path.join(d, str(f1))
                self.files.append(d2)
    
    def parse_file(self, path):
        with codecs.open(path, 'r', 'utf-8') as f:
            '''
            https://www.geeksforgeeks.org/beautifulsoup-scraping-paragraphs-from-html/
            '''
            dom =  BeautifulSoup(f.read(), 'html.parser')
        
        mini_path = self.get_mini_path(path)

        important_words = []
        for title in dom.find_all("title"):
            words = nltk.word_tokenize(title.get_text().strip())
            words = [w.lower() for w in words if w.isalnum()]
            for w in words:
                w = self.word_net_lemma.lemmatize(w)
                if w not in self.stop_words and w in self.english_valid_words:
                    important_words.append(w)
        self.doc_path_important_words_map[self.get_mini_path(path)] = important_words

        document = dom.get_text().strip()
        s = ''
        for i in document:
            if i.isascii() == False or i.isalnum() == False:
                s += ' '
            else:
                i = i.lower()
                s += i 
        words = nltk.word_tokenize(s)
        self.total_word_count = len(words)
        '''
        https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
        '''
        valid_words = []
        '''
        https://pythonspot.com/nltk-stop-words/
        '''
        for word in words:
            word = self.word_net_lemma.lemmatize(word)
            if word not in self.stop_words and word in self.english_valid_words:
                valid_words.append(word)

        self.doc_path_words_map[mini_path] = valid_words

        word_count_map = {}
        for w in valid_words:
            if w in word_count_map:
                word_count_map[w] += 1
            else:
                word_count_map[w] = 1
        self.doc_path_word_count_map[mini_path] = word_count_map

        return valid_words 
    
    def extract_tokens(self, path):
        mini_path = self.get_mini_path(path)
        words = self.parse_file(path)
        # word: {}
        result = dict()
        for word in words:
            if word in result:
                result[word]['num'] += 1
            else:
                result[word] = {'num': 1, 'tf': 0, 'idf': 0, 'doc_path': mini_path}

        for word in result:
            result[word]['tf'] = result[word]['num'] / self.total_word_count

            if word in self.word_docs_map.keys():
                self.word_docs_map[word].append(result[word])
            else:
                self.word_docs_map[word] = [result[word]]

        return result

    def fillout_idf(self):
        for word in self.word_docs_map:
            #print(i)
            doc_count = len(self.word_docs_map[word])
            idf = math.log(self.total_doc_count / float(doc_count))

            for j in self.word_docs_map[word]:
                j['idf'] = idf 
                j['tfidf'] = idf * j['tf']

    def extract_word_id(self):
        words = sorted(self.word_docs_map.keys())
        for i, w in enumerate(words):
            self.word_id_map[w] = i + 1

    def save(self):
        with open(WORD_DOCS_MAP_PATH, 'w') as f:
            json.dump(self.word_docs_map, f, indent=4)

        with open(DOC_PATH_WORDS_MAP_PATH, "w") as f:
            json.dump(self.doc_path_words_map, f, indent=4)
        
        with open(DOC_PATH_IMPORTANT_WORDS_MAP_PATH, "w") as f:
            json.dump(self.doc_path_important_words_map, f, indent=4)

        with open(WORD_ID_PATH, "w") as f:
            json.dump(self.word_id_map, f, indent=4)

        with open(DOC_PATH_WORD_COUNT_MAP_PATH, "w") as f:
            json.dump(self.doc_path_word_count_map, f, indent=4)
    
    @classmethod
    def get_mini_path(cls, path):
        p = Path(path)
        sep = p.parts
        return sep[-2] + '/' + sep[-1]

    def load_bookkeeping(self, work_dir):
        path = os.path.join(work_dir, "bookkeeping.json")
        with open(path) as f:
            self.data = json.load(f)
    
        
if __name__ == '__main__':
    d = 'WEBPAGES_RAW'
    ty = Database()
    ty.collect(d)
    

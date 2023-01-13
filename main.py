import json
import math 
import os 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np

from flask import Flask, render_template, url_for, request

from indexing import PATH_DOC_ID_MAP_PATH, DOC_ID_PATH_MAP_PATH, WORD_DOC_IDS_MAP_PATH, \
    DOC_PATH_WORDS_MAP_PATH, WORD_DOCS_MAP_PATH
from database_make import WORD_ID_PATH, DOC_PATH_WORD_COUNT_MAP_PATH, DOC_PATH_IMPORTANT_WORDS_MAP_PATH


from nltk.corpus import words as english_words



app = Flask(__name__)

engine = None


class Engine:
    def __init__(self):
        self.path_doc_id_map = {}
        with open(PATH_DOC_ID_MAP_PATH) as f:
            self.path_doc_id_map = json.load(f)

        self.doc_id_path_map = {}
        with open(DOC_ID_PATH_MAP_PATH) as f:
            self.doc_id_path_map = json.load(f)
        
        self.word_docs_map = {}
        with open(WORD_DOCS_MAP_PATH) as f:
            self.word_docs_map = json.load(f)
        print("word count {}".format(len(self.word_docs_map)))
        
        self.word_doc_ids_map = {}
        with open(WORD_DOC_IDS_MAP_PATH) as f:
            self.word_doc_ids_map = json.load(f)

        self.word_id_map = {}
        with open(WORD_ID_PATH) as f:
            self.word_id_map = json.load(f)

        self.doc_path_words_map = {}
        with open(DOC_PATH_WORDS_MAP_PATH) as f:
            self.doc_path_words_map = json.load(f)

        self.doc_path_word_count_map = {}
        with open(DOC_PATH_WORD_COUNT_MAP_PATH) as f:
            self.doc_path_word_count_map = json.load(f)

        self.doc_path_important_words_map = {}
        with open(DOC_PATH_IMPORTANT_WORDS_MAP_PATH) as f:
            self.doc_path_important_words_map = json.load(f)

        self.doc_path_link_map = {}
        with open('WEBPAGES_RAW/bookkeeping.json') as f:
            self.doc_path_link_map = json.load(f)

        self.total_word_count = len(self.word_id_map)
        self.total_doc_count = len(self.doc_id_path_map)

        self.stop_words = set(stopwords.words('english'))
        self.english_valid_words = set(english_words.words())
        self.word_net_lemma = WordNetLemmatizer()

    def search(self, query, max_count=50):
        tokens = self.extract_tokens(query)
        print("Query {}, tokens {}".format(query, tokens))
        candidate_doc_ids = []
        for token in tokens:
            doc_ids = self.word_doc_ids_map[token]
            print("token {}, doc ids {}".format(token, doc_ids))
            candidate_doc_ids += doc_ids

        query_vec = self.get_query_vector(query)
        print("query vector is {}".format(query_vec))

        candidate_doc_ids = list(set(candidate_doc_ids))
        print("candidate doc ids {}".format(len(candidate_doc_ids)))
        doc_scores = []  # [(doc_id, score)]
        for doc_id in candidate_doc_ids:
            doc_vec = self.get_doc_vector(doc_id)
            similarity = self.cosine(doc_vec, query_vec)
            print("doc_id {}, vec {}, similarity {}".format(doc_id, doc_vec, similarity))
            doc_scores.append((doc_id, similarity))

        top_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:max_count]
        result = []
        for doc_id, score in top_docs:
            info = self.get_doc_info(doc_id)
            info['score'] = score
            result.append(info)

        return result

    def get_doc_info(self, doc_id):
        path = self.doc_id_path_map[str(doc_id)]
        important_words = self.doc_path_important_words_map[path]
        result = {
            "path": path,
            "important_words": important_words,
            "url": self.doc_path_link_map[path],
        }
        
        return result
    
    def get_doc_vector(self, doc_id):
        doc_path = self.doc_id_path_map[str(doc_id)]
        word_count_map = self.doc_path_word_count_map[doc_path]
        
        vec = np.zeros(self.total_word_count)
        for word, count in word_count_map.items():
            word_id = self.word_id_map[word]
            tf = count / len(self.doc_path_words_map[doc_path])
            idf = math.log(self.total_doc_count / len(self.word_docs_map[word]))
            #print("word {}, tf {}, idf {}, count {}".format(word, tf, idf, count))
            vec[word_id - 1] = (1 + math.log(tf)) * idf

        return vec

    def extract_tokens(self, query):
        tokens = []
        words = nltk.word_tokenize(query.strip())
        words = [w.lower() for w in words if w.isalnum()]
        for w in words:
            w = self.word_net_lemma.lemmatize(w)
            if w not in self.word_id_map:
                continue
            if w not in self.stop_words and w in self.english_valid_words:
                tokens.append(w)
        
        return tokens

    def get_query_vector(self, query):
        tokens = self.extract_tokens(query)
        word_count_map = {}
        for token in tokens:
            if token in word_count_map:
                word_count_map[token] += 1
            else:
                word_count_map[token] = 1
        
        vec = np.zeros(self.total_word_count)
        for word, count in word_count_map.items():
            word_id = self.word_id_map[word]
            tf = count / len(tokens)
            idf = math.log(self.total_doc_count / len(self.word_docs_map[word]))
            #print("word {}, tf {}, idf {}, count {}".format(word, tf, idf, count))
            vec[word_id - 1] = (1 + math.log(tf)) * idf

        return vec

    def cosine(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))



@app.route("/", methods=["post", "get"])
def index():
    if request.method == "POST":
        query = request.form['query']
        result = engine.search(query)
        return render_template("index.html", query=query, result=result)
    else:
        return render_template("index.html")


if __name__ == '__main__':
    engine = Engine()
    #print(engine.search("Accessing Text Corpora and Lexical Resources"))
    app.run(debug=True)
    
    
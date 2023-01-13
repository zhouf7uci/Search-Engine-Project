""" Construct indexing files, output to Simple2.json
"""

import json
import os 
from database_make import Database, WORD_DOCS_MAP_PATH, DOC_PATH_WORDS_MAP_PATH, MAX_DIR_NO
from tqdm import tqdm


PATH_DOC_ID_MAP_PATH = "path_doc_id_map.json"
DOC_ID_PATH_MAP_PATH = "doc_id_path_map.json"
WORD_DOC_IDS_MAP_PATH = "word_doc_ids_map.json"

'''
Two classes of linklists are cite from 
https://medium.com/voice-tech-podcast/information-retrieval-using-boolean-query-in-python-e0ea9bf57f76 
'''
class Node:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.nextval = None


class LinkedList:
    def __init__(self, head = None):
        self.head = head
        self.tail = None
    
    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

    def append(self, node):
        if not self.head:
            self.head = node
            self.tail = node
            return

        tail = self.tail
        tail.nextval = node
        self.tail = node


class Retr:
    
    def __init__(self):
        self.path_doc_id_map = {}
        self.doc_id_path_map = {}
       
        self.doc_path_words_map = {}
        # load doc path words map
        with open(DOC_PATH_WORDS_MAP_PATH) as f:
            self.doc_path_words_map = json.load(f)
        
        self.word_docs_map = {}
        with open(WORD_DOCS_MAP_PATH) as f:
            self.word_docs_map = json.load(f)
        self.word_doc_ids_map = {}
    
    def start(self, work_dir):
        word_linked_list_map = {} 
        for word in self.word_docs_map.keys():
            word_linked_list_map[word] = LinkedList(None)
        
        doc_idx = 0
        for word in range(0, MAX_DIR_NO):
            d = os.path.join(work_dir, str(word))
            for f1 in os.listdir(d):
                d2 = os.path.join(d, f1)
                mini_path = Database.get_mini_path(d2)
                self.path_doc_id_map[mini_path] = doc_idx 
                self.doc_id_path_map[doc_idx] = mini_path
                doc_idx += 1 
        print(doc_idx)

        
        paths = list(self.path_doc_id_map.keys())
        for path in tqdm(paths):
            doc_id = self.path_doc_id_map[path]
            words = self.doc_path_words_map[path]
            # if not words:
            #    print("warning: words of {} is null".format(path))
            # else:
            #    #print("words of {} is {}".format(path, words))

            uniq_words = list(set(words))
            for word in uniq_words:
                word_linked_list_map[word].append(Node(doc_id))
        
        '''
        for loop below is cited from https://medium.com/voice-tech-podcast/information-retrieval-using-boolean-query-in-python-e0ea9bf57f76
        '''
        # word: [doc_id, ...]
        self.word_doc_ids_map = {}
        words = list(self.word_docs_map.keys())
        for word in tqdm(words):
            doc_ids = []
            linked_list = word_linked_list_map[word]
            header = linked_list.get_head()
            while header:
                if header.doc_id > 0:
                    doc_ids.append(header.doc_id)
                header = header.nextval
            self.word_doc_ids_map[word] = doc_ids
        
        with open(WORD_DOC_IDS_MAP_PATH, 'w') as f:
            json.dump(self.word_doc_ids_map, f, indent=4)

        with open(PATH_DOC_ID_MAP_PATH, "w") as f:
            json.dump(self.path_doc_id_map, f, indent=4)

        with open(DOC_ID_PATH_MAP_PATH, "w") as f:
            json.dump(self.doc_id_path_map, f, indent=4)
        
        

if __name__ == '__main__':
    d = 'WEBPAGES_RAW'
    ty = Retr()
    ty.start(d)

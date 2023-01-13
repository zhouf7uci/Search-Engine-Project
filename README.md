# Search-Engine-Project
Database_make.py parsed and processed the extracted tokens using the crawl-able HTML files inside the zip file, removed stop words from and applied lemmatization to the identified tokens.

Indexing.py constructed these tokens into an inverted index saved in a json file to improve the efficiency of search results. The inverted index is a simple map with the token as a key and a list of its corresponding postings, which contains the document name, word frequency, indices of occurrence within the document, and a Tf-IDF score.

Index.html is an HTML template for prompting the user for a query. At the time of the query, the main.py will look up indexes, perform calculations based on their TF-IDF score and importance of words in HTML tags, and give out a ranked list of relevant query pages for the query. Also, I added scoring mechanisms to help refine the search results.

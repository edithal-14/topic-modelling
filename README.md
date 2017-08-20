Topic Modelling project
=========================

ABOUT
--------------------------
This is my Internship project at Elsevier on section wise topic modelling of Medical corpus

AIM
--------------------------
Given a zip file of Medical corpus, output the 
topic term distribution and document topic distribuion in human readable format.

The Medicine zip file will be used for training and learning latent topics
from the corpora. The corpus can also be splitted for testing purposes.

BEFORE RUNNING THE PROJECT
------------------------------
Before running the project, ensure that the Medicine.zip file is in the same
directory as the scripts and the stopwords file

**Change the file /usr/local/lib/python3.5/dist-packages/sklearn/decomposition/nmf.py as follows**
* In line no. 1026 in the file mentioned above, **make update_H=False** instead of update_H=True
* You will need to have administrator access for doing this.
* Make sure this change is made before running the topic_learning_module

REQUIREMENTS
------------------------------
* NLTK (with stopwords corpora downloaded)
* ElementTree
* Sklearn
* Pickle
* Octave
* oct2py
* Symnmf2 directory, already provided in the project
* Java JDK should be installed
* Download the latest dbpedia spotlight jar file and the latest english model
 * have a look at: https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Run-from-a-JAR
 * Or use the already provided dbpedia-spotlight-0.7.1.jar (jar file) and en_2+2 (english model).

HOW TO RUN THE PROJECT
----------------------------
The main file which you want to have a look at is the topic_modelling.py,

The main script calls different modules/functions which constitute the basic blocks of the workflow.
You can call them seperately in a seperate script if you want (by importing those modules from the main script,
 remember to follow the same sequence as the main code while calling the modules), or you can simply run the topic_modelling.py script
which will automatically call the various modules in the appropriate sequence.

To know how to call the modules seperately have a look at the main function of
topic_modelling.py

OUTPUT
-----------------------
All the required directories will be created automatically, many pickle files, 
.mat files and .txt files will be created which are used by the various modules.
These created files can be further analyzed to gain important insight of the Medicine corpus


FUNCTIONING OF EACH MODULE
==============================

## 1. EXTRACTION MODULE

* Requirements: Medicine.zip file should exist in the present working directory (PWD)
 * Make sure that Medicine.zip has correct directory structure
 * Look at the directory structure of Medicine.zip
 * The zip file should have multiple zip files inside a folder called Medicine
* Input arguments: Nothing
* Output: A new directory called Medicine will be created in the PWD
* Function: Extracts the Medicine.zip file in the PWD

------------------------------

## 2. PARSING MODULE

* Requirements: Extracted directory called Medcine should exist in the PWD
* Input arguments: Nothing
* Output: A new directory calld text_corpus_full will be created containing the parsed .txt files
* Function: Parses the xmls in the Medicine directory by dividing a xml into sections and extracting all the text inside it. Also checks for english language

---------------------------------

## 3. START SERVER MODULE

* Requirements: Download the latest dbpedia spotlight jar file and the latest english model
 * have a look at: https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Run-from-a-JAR
 * Or use the already provided dbpedia-spotlight-0.7.1.jar (jar file) and en_2+2 (english model).
 * text_corpus_full directory should exist in the PWD
* Input arguments:
 *1. Path of the dbpedia spotlight JAR file
 *2. Path of the dbpedia spotlight english language model directory
* Output: Starts the server locally and returns the subprocess object of the server
* Function: Executes the bash command to start the dbpedia spotlight server

----------------------------------

## 4. CLEANING MODULE

* Requirements: Dbpedia spotlight server should be started using the Start server module
* Input arguments: The subprocess object of the server
* Output: Creates a directory called dbpedia_filtered_corpus_full **(TRAINING CORPUS)** containing the annotated .txt files
* Function: Calls the dbpedia spotlight servers annotate service, to get the dbpedia mentions from the text documents in text_corpus_full

-----------------------------------

## 5. SPLITTING MODULE

* Requirements: dbpedia_filtered_corpus_full directory should exist in the PWD
* Input arguments: Percentage of documents to be used for testing purpose after training
* Output: Creates a directory called testing_data **(TESTING CORPUS)** which contains the documents to be used for testing
* Function: Moves a given percentage of random documents from dbpedia_filtered_corpus_full to testing_directory

------------------------------------

## 6. FEATURE EXTRACTION MODULE

* Requirements: dbpedia_filtered_corpus_full directory should exist in the PWD
* Input arguments: Nothing
* Output: Creates a pickle file called features containing the important features of the **TRAINING CORPUS**
* Function: Calculates features like tfidf , document term occurence, vocabulary from the **TRAINING CORPUS**

-------------------------------------

## 7. TOPIC LEARNING MODULE

* Requirements: features file should exist in the PWD, symnmf2 directory is also needed.
* Input arguments: Path of the symnmf2 directory and No. of latent topics to learn
* Output: Creates 4 files, term_topic_<given_rank>, term_topic_details_<given_rank>.txt, doc_topic_<given_rank>, doc_topic_details_<given_rank>.txt
* Function:
 * **term_topic_(given_rank)** contains the term_topic matrix in matlab format
 * **term_topic_details_(given_rank).txt** has the reconstruction error after performing symmetric nmf
 * **doc_topic_(given_rank)** contains the document_topic matrix in pickle format
 * **doc_topic_details_(given_rank).txt** has details of the performance of NMF which does document topic inference

---------------------------------------

## 8. ANALYZE MODULE

* Requirements: Topic learning module should be executed for the given rank
* Input arguments: Rank of the term_topic and document_topic matrix to analyze
* Output: Creates three files, analyzed_term_topic_<given_rank>.txt, analyzed_doc_topic_<given_rank>.txt, coherence_term_topic_<given_rank>
* Function:
 * **analyzed_term_topic_(given_rank).txt** has the topic coherence statistics and the top 10 important terms in each topic
 * **analyzed_doc_topic_(given_rank).txt** has the top 3 important topics in each document
 * **coherence_term_topic_(given_rank)** is a pickle file containing a list of coherence of all the learnt topics

----------------------------------------

## 9. TESTING MODULE

* Requirements: testing_data directory should exist in PWD, Topic learning module should be executed for the given rank
* Input arguments: Rank of topic_term matrix to be used for inferring the topics of each document
* Output: Creates a file called **testing_doc_topic_(given_rank)**.txt
* Function: Calculates the top 3 topics for each of the testing documents

-----------------------------------------

My Contact details
==============================
Feel free to contact me for any queries.
* Email: vigneshedithal11031997v@gmail.com
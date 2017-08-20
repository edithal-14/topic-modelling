import zipfile
import os
import threading
from time import sleep
import sys
import time
import shutil
import codecs
from xml.dom import minidom
from xml.etree import ElementTree as ET
import random
import spotlight # pyspotlight
from nltk.corpus import stopwords #should have stopwords corpora downloaded
from functools import partial
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import scipy.sparse as sps
import scipy.io as sio
from sklearn.decomposition import NMF
import math
from oct2py import Oct2Py
# should have java installed, dbpedia jar file and model should be installed
# symnmf2 is needed, download it from http://math.ucla.edu/~dakuang/software/symnmf2.zip

def extract_member(zip,member,path):
	# extract the given member of a zip file
	zip.extract(member,path=path)
	
def extract_zip_files(files):
	# files: List of zip files to be extracted
	n_threads = 400 # these many simultaneous unzip process will be running
	for file in files:
		print("Extracting: "+file)
		zip_ref = zipfile.ZipFile(file,"r")
		members = zip_ref.namelist()
		for member in members:
			if not os.path.exists(file[:-4]+"/"+"/".join(member.split("/")[:-1])+"/"):
				os.makedirs(file[:-4]+"/"+"/".join(member.split("/")[:-1])+"/")
			while threading.activeCount() >= n_threads:
				sleep(2)  # wait for no. of active threads to decrease to n_threads-1
			thread = threading.Thread(target=extract_member,args=(zip_ref,member,file[:-4]+"/"))
			thread.start()

def extraction_module():
	# extract all the zip files in the present directory
	cwd = os.getcwd()+"/"
	zip_file_names = [cwd+i for i in os.listdir(cwd) if i.endswith(".zip") or i.endswith(".ZIP")]
	extract_zip_files(zip_file_names)
	while threading.activeCount() >= 2:
		sleep(2) # wait for no. of active threads to decrease to 1
	# moving files form Medicine/Medcine/ to Medicine/
	for file in os.listdir("Medicine/Medicine/"):
		shutil.move("Medicine/Medicine/"+file,"Medicine/"+file)
	# deleting Medicine/Medcine/
	os.rmdir("Medicine/Medicine/")
	for zip_file_name in zip_file_names:
		dir = zip_file_name[:-4]+"/"
		zip_file_names1 = [dir+i for i in os.listdir(dir) if i.endswith(".zip") or i.endswith(".ZIP")]
		extract_zip_files(zip_file_names1)
		for zip_file_name1 in zip_file_names1:
			os.remove(zip_file_name1)

def parse(dirpath,corpus_dir,file):
	zip_name = dirpath.split("/")[-3]
	isbn = dirpath.split("/")[-2]
	chapter_id = dirpath.split("/")[-1]
	xml_path = dirpath+"/"+file
	namespaces = {"ce":"http://www.elsevier.com/xml/common/schema", 'rawtext':'http://www.elsevier.com/xml/common/doc-properties/schema','bk':'http://www.elsevier.com/xml/bk/schema'}
	with open(xml_path, "r") as xmlFile:
		tree = ET.parse(xmlFile)
	root = tree.getroot()
	section_index = 1
	# check for language
	chapter_tags = root.findall('.//bk:chapter',namespaces)
	simple_chapter_tags = root.findall('.//bk:simple-chapter',namespaces)
	lang_key = '{http://www.w3.org/XML/1998/namespace}lang'
	if len(chapter_tags)>0:
		if chapter_tags[0].attrib[lang_key]!='en':
			return # dont want this
	elif len(simple_chapter_tags)>0:
		if simple_chapter_tags[0].attrib[lang_key]!='en':
			return # dont want this
	rawtextNodes = root.findall('.//rawtext:raw-text',namespaces)
	sectionsNodes = root.findall(".//ce:sections", namespaces)
	for rawtextNode in rawtextNodes:
		with codecs.open(corpus_dir+isbn+"_"+chapter_id+"_"+str(section_index)+".txt","w","utf-8") as wf:
			wf.write(" ".join(rawtextNode.itertext()))
			section_index+=1
	for sectionsNode in sectionsNodes:
		para_text = ""
		for node in sectionsNode:
			if node.tag.endswith("para"):
				para_text=para_text+" ".join(node.itertext())+" "
			elif node.tag.endswith("section"):
				if len(para_text)>0:
					with codecs.open(corpus_dir+zip_name+"_"+isbn+"_"+chapter_id+"_"+str(section_index)+".txt","w","utf-8") as wf:
						wf.write(para_text.strip())
					section_index+=1
					para_text=""
				with codecs.open(corpus_dir+zip_name+"_"+isbn+"_"+chapter_id+"_"+str(section_index)+".txt","w","utf-8") as wf:
					wf.write(" ".join(node.itertext()))
				section_index+=1
		if len(para_text)>0:
			with codecs.open(corpus_dir+zip_name+"_"+isbn+"_"+chapter_id+"_"+str(section_index)+".txt","w","utf-8") as wf:
				wf.write(para_text.strip())
				
def crawl(path,corpus_dir):
	n_threads=400 # these many simultaneous parsing processes will run
	for (dirpath, dirnames, files) in os.walk(path):
		for file in files:
			if file.endswith(".xml"):
				while threading.activeCount() >= n_threads:
					sleep(2)  # wait for no. of active threads to decrease to n_threads-1
				thread = threading.Thread(target=parse,args=(dirpath,corpus_dir,file))
				thread.start()
	while threading.activeCount() >= 2:
		sleep(2) # wait for no. of threads to decrease to 1

def parsing_module():
	main_dir=os.getcwd()+"/"
	corpus_dir = main_dir+"text_corpus_full/"
	if not os.path.exists(corpus_dir):
		os.makedirs(corpus_dir)
	crawl(main_dir+"Medicine/",corpus_dir)

def clean(path,file,api,dir):
	with codecs.open(path + "text_corpus_full/" + file, "r", "utf-8") as f:
		content = f.read()
	try:
		custom_filter = {'policy':"whitelist",'types':"DBpedia:AnatomicalStructure, DBpedia:ChemicalSubstance, DBpedia:Database, DBpedia:Device, DBpedia:Disease, DBpedia:Drug, DBpedia:EthnicGroup, DBpedia:Food, DBpedia:Protein, DBpedia:Species, Freebase:/biology, Freebase:/chemistry, Freebase:/food, Freebase:/medicine, Freebase:/physics, Schema:Product",'coreferenceResolution': False}
		annotation = api(content,filters=custom_filter)
		words=[]
		for ann in annotation:
			if ('surfaceForm' in ann.keys()):
				if not ( type(ann['surfaceForm'])==int or type(ann['surfaceForm'])==float ):
					words.append(ann['surfaceForm']+"\t"+ann['URI'])
		cleaned_text = "\n".join(words)
		if len(cleaned_text) > 0:
			with codecs.open(path+dir+file,"w","utf-8") as f:
				f.write(cleaned_text)
	except spotlight.SpotlightException as e:
		print("File: "+file+" ,Dbpedia could not extarct anything useful")

def cleaning_module(process):
	path = os.getcwd()+"/"
	dir= "dbpedia_filtered_corpus_full/"
	if not os.path.exists(path+dir):
		os.makedirs(path+dir)
	api = partial(spotlight.annotate,"http://localhost:2222/rest/annotate",confidence=0,support=0)
	files = os.listdir(path+"text_corpus_full")
	#if the cleaning module has been run before and it is re runned again, 
	#already cleaned documents will not be processed again and ignored.
	done=[]
	for file in os.listdir(path+dir):
		#these files have already been processed
		done.append(file)
	n_threads=16 # these many simultaneous cleaning operations will take place
	for file in files:
		if file in done:
			#ignore files which are already dbpedia cleaned
			continue
		while threading.activeCount()>=n_threads:
			sleep(1) # sleep for 1 second
		thread = threading.Thread(target=clean,args=(path,file,api,dir))
		thread.start()
	while threading.activeCount() >= 2:
		sleep(2) # wait for no. of threads to decrease to 1
	process.terminate() # kill the dbpedia spotlight server
	#removing stopwords annotated by dbpedia spotlight from each file		
	with open(path+"lextek_stopwords.txt","r") as f:
		lextek_stopwords = f.read().split()
	stop_words=set(stopwords.words("english")+lextek_stopwords+["and/or"])
	for file in os.listdir(path+dir):
		good=[]
		with open(path+dir+file,"r") as f:
			lines = f.read().splitlines()
		for line in lines:
			if line.split("\t")[0].lower() not in stop_words:
				good.append(line)
		with open(path+dir+file,"w") as f:
			for line in good:
				f.write(line+"\n")
	#removing files which are empty after removal of stopwords
	for file in os.listdir(path+dir):
		with open(path+dir+file,"r") as f:
			lines = f.read().splitlines()
		if len(lines)==0:
			os.remove(path+dir+file)

def feature_extraction_module():
	dir= os.getcwd()+"/dbpedia_filtered_corpus_full/"
	files = [dir+file for file in os.listdir(dir)]
	custom_tokenizer = lambda text:[line.split("\t")[1] for line in text.splitlines()]
	cv = CountVectorizer(input="filename",tokenizer=custom_tokenizer,min_df=2)
	doc_term = cv.fit_transform(files)
	# this vocab. contains lower case uris
	vocab = cv.vocabulary_
	custom_pre = lambda text:text # this is custom preprocessor
	# this vocab contains original uris
	vocab_new = CountVectorizer(input="filename",tokenizer=custom_tokenizer,min_df=2,preprocessor=custom_pre).fit(files).vocabulary_
	# this is a mapping between lower case uri to original uri
	mapping = dict([[key.lower(),key] for key in vocab_new.keys()])
	vocab_dict = dict([(vocab[key],key) for key in vocab.keys()])
	tfidf_vec = TfidfVectorizer(input="filename",tokenizer=custom_tokenizer,min_df=2,sublinear_tf = True)
	tfidf = tfidf_vec.fit_transform(files)
	doc_term_oc = doc_term > 0
	doc_term_oc = doc_term_oc.astype('int')
	term_co = np.dot(doc_term_oc.T,doc_term_oc)
	term_oc = np.sum(doc_term_oc,axis=0).tolist()[0]
	sum_term_co = term_co.sum()
	sum_col_term_co = np.sum(term_co,axis=0).tolist()[0]
	row,col = term_co.nonzero()
	non_zero = [(row[k],col[k]) for k in range(len(row))]
	term_co_ppmi = np.zeros((term_co.shape[0],term_co.shape[1]))
	term_co_lil = term_co.tolil()
	for i,j in non_zero:
		term_co_ppmi[i,j] = max(math.log((term_co_lil[i,j]/float(sum_term_co))/float((sum_col_term_co[i]/float(sum_term_co))*(sum_col_term_co[j]/float(sum_term_co)))),0)
	term_co_ppmi = sps.csr_matrix(term_co_ppmi)
	term_corelation = cosine_similarity(term_co_ppmi,dense_output=False)
	# save the extracted features
	with open("features","wb") as f:
		pickle.dump([doc_term,tfidf,vocab_dict,files,mapping],f)
	# save term_corelation matrix in matlab format to run symmetric nmf on it
	term_corelation = term_corelation.toarray()
	# split matrix into chunks of 1000 rows
	chunks = np.split(term_corelation,[i for i in range(2000,term_corelation.shape[0],2000)])
	save_dict=dict([("part"+str(chr(i+65)),chunks[i]) for i in range(len(chunks))])
	# save the splitted matrix
	sio.savemat("term_corelation",save_dict)

def topic_learning(path,tar_rank):
	oc = Oct2Py()
	tfidf = pickle.load(open("features","rb"))[1]
	start_time=time.time()
	oc.symnmf(path,str(tar_rank))
	U = sio.loadmat("term_topic_"+str(tar_rank))['U']
	nmf = NMF(n_components=int(tar_rank),init="custom",verbose=2)
	V = nmf.fit_transform(tfidf,H=U.T)
	# since V can be large , use protocol=4
	with open("doc_topic_"+str(tar_rank),"wb") as f:
		pickle.dump(V,f,protocol=4)
	error = nmf.reconstruction_err_
	with open("doc_topic_details_"+str(tar_rank)+".txt","w") as f:
		f.write("Reconstruction_error: "+str(error))
	end_time = time.time()
	print("Learning topics and document topic inference done!!\nTime taken (seconds): "+str(end_time-start_time)+" for "+str(tar_rank)+" latent topics")

def start_server(jar_path,model_path):
	command = "java -jar "+jar_path+" "+model_path+" http://localhost:2222/rest"
	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	sleep(120) # wait 2 mins for server to start
	return process

def splitting_module(percent):
	dir= os.getcwd()+"/"
	files = os.listdir(dir+"dbpedia_filtered_corpus_full/")
	testing_docs = np.random.choice(files,int(len(files)*percent/100),replace=False)
	if not os.path.exists(dir+"testing_data/"):
		os.makedirs(dir+"testing_data/")
	for file in testing_docs:
		shutil.move(dir+"dbpedia_filtered_corpus_full/"+file,dir+"testing_data/"+file)

def testing_module(rank):
	# get the feature of the unseen documents like tfidf
	dir= os.getcwd()+"/testing_data/"
	files = [dir+file for file in os.listdir(dir)]
	custom_tokenizer = lambda text:[line.split("\t")[1] for line in text.splitlines()]
	tfidf_vec = TfidfVectorizer(input="filename",tokenizer=custom_tokenizer,min_df=2,sublinear_tf = True)
	tfidf = tfidf_vec.fit_transform(files)
	vocab = tfidf_vec.vocabulary_
	# using this tfidf matrix, build a new tfidf matrix by matching the column of tfidf matrix with the learnt vocabulary, in case of no match the column is all zeros
	training_vocab_dict = pickle.load(open("features","rb"))[2]
	columns = []
	for key in training_vocab_dict.keys():
		try:
			columns.append(tfidf.getcol(vocab[training_vocab_dict[key]]))
		except KeyError:
			columns.append(sps.csr_matrix(np.zeros((tfidf.shape[0],1))))
	tfidf = sps.hstack(columns)
	# load the trained U (term_topic) and get the V (doc_topic) matrix using nmf
	U = sio.loadmat("term_topic_"+str(rank))['U']
	nmf = NMF(n_components=int(rank),init="custom",verbose=2)
	V = nmf.fit_transform(tfidf,H=U.T)
	# since V can be large , use protocol=4
	with open("testing_doc_topic_"+str(rank),"wb") as f:
		pickle.dump(V,f,protocol=4)
	# analyze the V matrix
	analyze_doc_topic(V,files,testing=True)

def analyze(rank):
	# load the features and the trained matrices
	[doc_term,tfidf,vocab_dict,files,mapping] = pickle.load(open("features","rb"))
	V = pickle.load(open("doc_topic_"+str(rank),"rb"))
	U = sio.loadmat("term_topic_"+str(rank))['U']
	# U is term_topic matrix
	analyze_term_topic(U,vocab_dict,doc_term,mapping)
	# V is doc_topic matrix
	analyze_doc_topic(V,files)

def analyze_term_topic(U,vocab_dict,doc_term,mapping):
	# get the term co-occurence matrix for calculating coherence
	doc_term_oc = doc_term > 0
	doc_term_oc = doc_term_oc.astype('int')
	term_co_dense = np.dot(doc_term_oc.T,doc_term_oc).todense()
	term_oc = np.sum(doc_term_oc,axis=0).tolist()[0]
	# given a column of term_topic matrix, identify row numbers with highest values
	topics_data = [[[tup[0],tup[1]] for tup in sorted(enumerate([U[j][i] for j in range(U.shape[0])]),key=lambda a:a[1],reverse=True)[:10]] for i in range(U.shape[1])]
	# calculate coherence for each topic based on the term co-occurence top 10 term
	topics_coherence = [sum([math.log((term_co_dense[i[0],j[0]]+(10**-12))/float(term_oc[i[0]])) for i in li for j in li]) for li in topics_data]
	topics_coherence = sorted(enumerate(topics_coherence),key= lambda a:a[1],reverse=True)
	avg_coherence = sum([tup[1] for tup in topics_coherence])/float(len(topics_coherence))
	best_coherence = topics_coherence[0][1]
	worst_coherence = topics_coherence[len(topics_coherence)-1][1]
	# print the results
	with open("analyzed_term_topic_"+str(U.shape[1])+".txt","w") as f:
		f.write("No. of topics: "+str(U.shape[1])+"\n")
		f.write("Best topic coherence: "+str(best_coherence)+"\n")
		f.write("Worst topic coherence: "+str(worst_coherence)+"\n")
		f.write("Average topic coherence: "+str(avg_coherence)+"\n\n")
		for i in [tup for tup in topics_coherence]:
			f.write("Topic no.: "+str(i[0])+"\n")
			f.write("Topic coherence: "+str(i[1])+"\n")
			f.write("\n".join([str(j+1)+". "+mapping[vocab_dict[topics_data[i[0]][j][0]]]+" relevance: "+str(topics_data[i[0]][j][1])+" " for j in range(len(topics_data[i[0]]))])+"\n\n")

def analyze_doc_topic(V,files,testing=False):
	# name should be different for results testing documents
	if testing:
		name = "analyzed_testing_doc_topic_"
	else:
		name = "analyzed_doc_topic_"
	# for each document print top 10 topics
	with open(name+str(V.shape[1])+".txt","w") as f:
		for i in range(V.shape[0]):
			f.write("File name: "+files[i]+"\n")
			# from a given row of doc_topic matrix , find the columns with the highest values and print column no. of those columns and their relevance value
			f.write("Topic no.: "+",".join([str(j[0])+" relevance: "+str(j[1])+" " for j in [tup for tup in sorted(enumerate([V[i][j] for j in range(V.shape[1])]),key=lambda a:a[1],reverse=True)[:3]]])+"\n\n")

if __name__=="__main__":
	# this will extract the Medicine.zip file in the current directory
	extraction_module()
	# this will parse the xmls in the extracted directory into sections
	parsing_module()
	# give the path of jar file and model directory to start the dbpedia spotlight server
	process = start_server("/home/ubuntu/internship/dbpedia-spotlight-0.7.1.jar","/home/ubuntu/internship/en_2+2")
	# this will use the dbpedia spotlight server to get the annotations
	cleaning_module(process)
	# this will move 10% documents from dbpedia_filtered_corpus directory
	# to testing_data directory, for testing purposes
	splitting_module(10)
	# give the path which contains dbpedia cleaned documents
	feature_extraction_module()
	# this for loop can be parallelized, this will help in getting the result
	# for different ranks (no. of topics) simultaneously. The analyzed files
	# can then be evaluated to find the best value of the desired no. of topics
	for rank in range(500,3000,500):
		# provide the path of symnmf2 directory and the desired no. of topics to learn
		topic_learning("/home/ubuntu/internship/symnmf2/",rank)
		# analyze the learnt term_topic and doc_topic matrices of the given rank
		# make sure these matrices are already trained before analyzing them
		analyze(rank)
		# this will infer the topics for documents in testing_data directory
		# provide the rank of the term_topic matrix which you want to use for testing
		# make sure term_topic_<desired_rank> is already trained and present in the
		# main directory
		testing_module(rank)
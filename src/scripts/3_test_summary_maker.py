
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hashlib import sha256
from datetime import datetime

from src.classes.profittm.TopicInterpreter import TopicInterpreter
from src.classes.utils import load, save
from src.classes.paths_config import *
from src.classes.profittm.DataBaseManager import DataBaseManager
from src.classes.profittm.SummaryMakerNN import SummaryMakerNN
np.random.seed(45)

#########################################################
# thematize texts

database_manager = DataBaseManager( database_path )
parsed_data = database_manager.get_parsed_data()
parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
parsed_data = parsed_data[ parsed_data["parse_datetime"] >= datetime(day=14, month=12, year=2022) ]
test_texts = parsed_data["content"].to_numpy()

vectorizer = load(os.path.join( interim_dir, "prod_vectorizer.pkl"))
text_vectors = vectorizer.vectorize_docs(test_texts, use_tfidf=True, n_jobs=8)

topic_model = load( os.path.join( interim_dir, "prod_tree_profittm.pkl") )
pred_y = topic_model.predict( text_vectors, return_vectors = False )
save( pred_y, os.path.join(interim_dir, "prod_topic_labels.pkl") )
topic_labels = load( os.path.join(interim_dir, "prod_topic_labels.pkl") )

###########################################
# stratified by low level topics subsample for debug purpose
"""level_group_hashes = []
for group in range( len(topic_labels) ):
    current_hashes = []
    current_group = ""
    for level in range( len(topic_labels[group]) ):
        
        if level == 0:
            current_group += str(topic_labels[group][level])
        else:
            current_group += "_" + str(topic_labels[group][level])
        current_group_hash = current_group
        
        #current_group += str(topic_labels[group][level]) + "_"
        #current_group_hash = current_group.encode("utf-8")
        #current_group_hash = sha256( current_group_hash )
        #current_group_hash = current_group_hash.hexdigest()
        
        current_hashes.append( current_group_hash )
    level_group_hashes.append( current_hashes )
level_group_hashes = np.array( level_group_hashes )


np.random.seed( 45 )
texts_subsample = []
labels_subsample = []
group_size = 20
uniq_low_level_hashes = np.unique( level_group_hashes[:, -1] )
for uniq_hash in uniq_low_level_hashes:
    selected_texts = test_texts[ level_group_hashes[:, -1] == uniq_hash]
    selected_labels = topic_labels[ level_group_hashes[:, -1] == uniq_hash]
    
    if len( selected_texts ) <= group_size:
        texts_subsample.append( selected_texts )
        labels_subsample.append( selected_labels )
        continue
    
    row_ids = [i for i in range( len(selected_texts) )]
    row_ids_subsample = np.random.choice( row_ids, size=group_size, replace=False )
    selected_texts = selected_texts[ row_ids_subsample ]
    selected_labels = selected_labels[ row_ids_subsample ]
    texts_subsample.append( selected_texts )
    labels_subsample.append( selected_labels )
    
texts_subsample = np.hstack( texts_subsample )
labels_subsample = np.vstack( labels_subsample )"""
#######


######################################
#np.random.seed(45)
#row_ids = [i for i in range( len(test_texts) )]
#row_ids_subsample = np.random.choice( row_ids, size=100, replace=False )
#texts_subsample = [ test_texts[i] for i in row_ids_subsample ]
######################################

summary_maker = SummaryMakerNN(device="cuda")
#text_summaries = load( Path( interim_dir, "prod_summaries.pkl" ) )
text_summaries, group_summaries = summary_maker.make_hierarchical_summaries( test_texts, topic_labels, suppression_rate=10, random_state=45, verbose=True, ready_summaries=None )
save( text_summaries, Path( interim_dir, "prod_summaries.pkl" ) )
save( group_summaries, Path( interim_dir, "prod_group_summaries.pkl" ) )

text_summaries = load( Path( interim_dir, "prod_summaries.pkl" ) )
group_summaries = load( Path( interim_dir, "prod_group_summaries.pkl" ) )

print("done")
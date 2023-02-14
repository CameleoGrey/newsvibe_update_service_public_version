
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from pprint import pprint
from nltk.tokenize import sent_tokenize

from src.classes.utils import *
from src.classes.paths_config import *

from src.classes.profittm.ContentScrapper import ContentScrapper
from src.classes.profittm.DataBaseManager import DataBaseManager

from src.classes.profittm.TreeProfitTM import TreeProfitTM
from src.classes.profittm.TfidfW2vVectorizer import TfidfW2vVectorizer

from src.classes.profittm.SummaryMakerNN import SummaryMakerNN
from src.classes.profittm.TopicInterpreter import TopicInterpreter

class UpdateContentPipeline():
    def __init__(self):
        
        self.start_update_time = datetime.utcnow()
        self.end_update_time = None
        
        pass
    
    def update_parsed_data(self, proxies=None, use_default_urls=False):
        
        
        database_manager = DataBaseManager( database_path )
        content_scrapper = ContentScrapper()
        
        if use_default_urls:
            urls_for_parsing = ContentScrapper()
        else:
            urls_for_parsing = content_scrapper.get_custom_urls( Path(production_dir, "valid_source_urls.csv") )
        urls_to_ignore = database_manager.get_urls_to_ignore()
        
        # debug
        #urls_for_parsing = urls_for_parsing[4:8]
        
        fresh_parsed_data, fresh_trash_urls = content_scrapper.parse_fresh_data(url_list=urls_for_parsing, 
                                                                                     sleep_time=0.2, 
                                                                                     verbose=True, 
                                                                                     n_jobs=10, 
                                                                                     urls_to_ignore=urls_to_ignore, 
                                                                                     proxies=proxies)
        database_manager.update_parsed_data( fresh_parsed_data )
        database_manager.update_trash_only_urls( fresh_trash_urls )
        
        pass
    
    def retrain_topic_model(self, fit_vectorizer_on_fresh_texts=True):
        
        database_manager = DataBaseManager( database_path )
        parsed_data = database_manager.get_parsed_data()
        parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
        all_parsed_data = parsed_data
        fresh_parsed_data = parsed_data[ parsed_data["parse_datetime"] >= self.start_update_time ]

        train_texts = np.unique( all_parsed_data["content"].to_numpy() )
        fresh_texts = np.unique( fresh_parsed_data["content"].to_numpy() )
        np.random.seed(45)
        np.random.shuffle( train_texts )
        
        vectorizer = TfidfW2vVectorizer()
        texts_for_vectorizer = None
        if fit_vectorizer_on_fresh_texts:
            texts_for_vectorizer = fresh_texts
        else:
            texts_for_vectorizer = train_texts
        vectorizer.fit(texts_for_vectorizer, vector_size=384, window=5,
                    n_jobs=10, min_count=2, sample=1e-5, epochs=100, sg=0, seed=45)
        save(vectorizer, os.path.join( interim_dir, "prod_vectorizer.pkl"))
        
        vectorizer = load(os.path.join( interim_dir, "prod_vectorizer.pkl"))
        vectorized_texts = vectorizer.vectorize_docs(fresh_texts, use_tfidf=True, n_jobs=8)
        save(vectorized_texts, os.path.join( interim_dir, "vectorized_fresh_texts.pkl"))
        
        vectorized_texts = load(os.path.join( interim_dir, "vectorized_fresh_texts.pkl"))
        topic_model = TreeProfitTM( max_depth=2 )
        topic_model.fit(vectorized_texts)
        save( topic_model , os.path.join( interim_dir, "prod_tree_profittm.pkl"), verbose=True)
        
        pass
    
    def make_fresh_summaries(self):
        database_manager = DataBaseManager( database_path )
        parsed_data = database_manager.get_parsed_data()
        parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
        parsed_data = parsed_data[ parsed_data["parse_datetime"] >= self.start_update_time ]
        test_texts = parsed_data["content"].to_numpy()
        
        vectorizer = load(os.path.join( interim_dir, "prod_vectorizer.pkl"))
        text_vectors = vectorizer.vectorize_docs(test_texts, use_tfidf=True, n_jobs=8)
        
        topic_model = load( os.path.join( interim_dir, "prod_tree_profittm.pkl") )
        pred_y = topic_model.predict( text_vectors, return_vectors = False )
        save( pred_y, os.path.join(interim_dir, "prod_topic_labels.pkl") )
        topic_labels = load( os.path.join(interim_dir, "prod_topic_labels.pkl") )
        
        summary_maker = SummaryMakerNN(device="cuda")
        #text_summaries = load( Path( interim_dir, "prod_summaries.pkl" ) )
        text_summaries, group_summaries = summary_maker.make_hierarchical_summaries( test_texts, topic_labels, suppression_rate=10, random_state=45, verbose=True, ready_summaries=None )
        save( text_summaries, Path( interim_dir, "prod_summaries.pkl" ) )
        save( group_summaries, Path( interim_dir, "prod_group_summaries.pkl" ) )
        
        pass
    
    def plot_useful_graphics(self):
        database_manager = DataBaseManager( database_path )
        parsed_data = database_manager.get_parsed_data()
        parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
        parsed_data = parsed_data[ parsed_data["parse_datetime"] >= self.start_update_time ]
        test_texts = parsed_data["content"].to_numpy()
        
        vectorizer = load(os.path.join( interim_dir, "prod_vectorizer.pkl"))
        test_raw_features = vectorizer.vectorize_docs(test_texts, use_tfidf=True, n_jobs=8)
        topic_interpreter = TopicInterpreter()
        topic_interpreter.vectorizer = vectorizer
        
        test_labels = load( os.path.join(interim_dir, "prod_topic_labels.pkl") )
        topic_model = load( os.path.join( interim_dir, "prod_tree_profittm.pkl") )
        
        
        # draw clusters
        #test_topic_features = topic_model.extract_features( test_raw_features)
        #print(np.isnan(test_topic_features).sum())
        #topic_interpreter.plot_clusters(test_topic_features, test_labels, level=0, n_jobs=10, plot_path=Path(images_dir, "clusters_0.jpg"))
        #topic_interpreter.plot_clusters(test_topic_features, test_labels, level=1, n_jobs=10, plot_path=Path(images_dir, "clusters_1.jpg"))
        #topic_interpreter.plot_clusters(test_topic_features, test_labels, level=None, n_jobs=10, plot_path=Path(images_dir, "clusters_2.jpg"))
        
        # draw distances
        #topic_interpreter.draw_distances(test_topic_features, test_labels, level=0, plot_path=Path(images_dir, "distances_0.jpg"))
        
        # extract topics
        topic_names_0 = topic_interpreter.get_topic_names(test_texts, test_labels, level=0, n_jobs=8)
        with open(Path(images_dir, "topic_names_0.json"), "w") as topic_names_file:
            json.dump(topic_names_0, topic_names_file)
        topic_names_1 = topic_interpreter.get_topic_names(test_texts, test_labels, level=1, n_jobs=8)
        with open(Path(images_dir, "topic_names_1.json"), "w") as topic_names_file:
            json.dump(topic_names_1, topic_names_file)
        
        # draw topic tree
        group_summaries = load( Path( interim_dir, "prod_group_summaries.pkl" ) )
        topic_interpreter.plot_topic_graph(test_texts, test_labels, 
                                           path_to_save = os.path.join(images_dir, "topic_graph.gv"), 
                                           n_jobs=8, group_summaries=group_summaries)
        
        pass
    
    def build_fresh_frontend_content(self):
        
        content_scrapper = ContentScrapper()
        urls_for_parsing = content_scrapper.get_custom_urls( Path(production_dir, "valid_source_urls.csv") )
        sources_count = len( urls_for_parsing )
        
        database_manager = DataBaseManager( database_path )
        parsed_data = database_manager.get_parsed_data()
        parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
        parsed_data = parsed_data[ parsed_data["parse_datetime"] >= self.start_update_time ]
        test_urls = parsed_data["url"].to_numpy()
        test_labels = load( os.path.join(interim_dir, "prod_topic_labels.pkl") )
        text_summaries = load( Path( interim_dir, "prod_summaries.pkl" ) )
        group_summaries = load( Path( interim_dir, "prod_group_summaries.pkl" ) )
        
        group_summaries = { k.replace(".0", ""): v for k, v in group_summaries.items() }
        
        article_summaries = np.vstack([test_urls, text_summaries]).T
        article_summaries = np.hstack([article_summaries, test_labels])
        article_summaries = article_summaries.T # transpose for faster frontend processing
        article_summaries[2] = article_summaries[2].astype(np.int64)
        article_summaries[3] = article_summaries[3].astype(np.int64)
        
        col_ids = [i for i in range(article_summaries.shape[1])]
        np.random.seed(45)
        np.random.shuffle( col_ids )
        article_summaries = article_summaries[:, col_ids]
        
        
        article_summaries = list(article_summaries)
        for i in range( len(article_summaries) ):
            article_summaries[i] = list( article_summaries[i] )
            
        # keywords generation
        topic_names = None
        with open(Path(images_dir, "topic_names_0.json"), "r") as topic_names_file:
            topic_names = json.load(topic_names_file)
        keywords = []
        keywords.append( "news vibe" )
        keywords.append( "newsvibe" )
        keywords.append( "news" )
        for topic_name in topic_names.keys():
            keywords_batch = topic_names[ topic_name ]
            keywords_batch = keywords_batch.split(" ")
            for word in keywords_batch:
                keywords.append( word )
        keywords = ", ".join( keywords )
        
        data_for_frontend = {}
        data_for_frontend["article_summaries"] = list(article_summaries)
        data_for_frontend["topic_summaries"] = group_summaries
        data_for_frontend["news_count"] = len(article_summaries[0])
        data_for_frontend["sources_count"] = sources_count
        data_for_frontend["update_time"] = str(self.start_update_time.strftime("%Y-%m-%d %H:%M:%S"))
        data_for_frontend["keywords"] = keywords
        
        def build_short_global_overview( group_summaries, sentences_per_topic ):
            
            high_level_topic_ids = []
            for topic_id in group_summaries.keys():
                if "_" not in topic_id:
                    high_level_topic_ids.append( topic_id )
            
            shorted_topic_summaries = []
            for topic_id in high_level_topic_ids:
                group_summary = group_summaries[topic_id]
                #summary_sentences = group_summary.split(".")
                summary_sentences = sent_tokenize(group_summary, language="english")
                summary_sentences = summary_sentences[:sentences_per_topic]
                short_summary = " ".join( summary_sentences )
                shorted_topic_summaries.append( short_summary )
            short_global_overview = "\n".join( shorted_topic_summaries )
                
            return short_global_overview
        
        data_for_frontend["short_global_overview"] = build_short_global_overview( group_summaries, sentences_per_topic=3 )
        
        def build_links_for_short_overview( article_summaries, links_per_topic = 10 ):
            links = np.array(article_summaries[0])
            descriptions = np.array(article_summaries[1])
            high_topic_ids = np.array(article_summaries[2])
            
            summary_links_data = []
            uniq_topic_ids = np.unique( high_topic_ids )
            for high_topic_id in uniq_topic_ids:
                current_topic_links = links[ high_topic_ids == high_topic_id ].copy()
                current_topic_summaries = descriptions[ high_topic_ids == high_topic_id ].copy()
                
                content_ids = np.array([i for i in range(len(current_topic_links))])
                np.random.seed(high_topic_id)
                np.random.shuffle( content_ids )
                overview_sample_ids = content_ids[:links_per_topic]
                
                current_topic_links = current_topic_links[ overview_sample_ids ]
                current_topic_summaries = current_topic_summaries[ overview_sample_ids ]
                
                for i in range( len(current_topic_links) ):
                    summary_links_data.append( [current_topic_links[i], current_topic_summaries[i]] )
            
            return summary_links_data
        
        data_for_frontend["short_global_links"] = build_links_for_short_overview( article_summaries, links_per_topic = 100 )
        
        with open(Path(production_dir, "data_for_frontend.json"), "w") as data_for_frontend_file:
            json.dump(data_for_frontend, data_for_frontend_file)
        
        self.end_update_time = datetime.utcnow()
        
        pass
    
        
    def send_update(self):
        
        content_full_path = Path(production_dir, "data_for_frontend.json")
        with open( content_full_path, "rb" ) as content_update_file:
            

            valid_api_keys = []
            api_keys_path = Path(production_dir, "valid_api_keys.txt")
            with open( api_keys_path, "r" ) as api_keys_file:
                for key_string in api_keys_file:
                    valid_api_keys.append( key_string )
            api_key = valid_api_keys[0]
            
            headers = {"Content-Type": "application/json",
                       "api-key": api_key}
            #response = requests.post("http://127.0.0.1:5000/fresh_content",
            #                         data=content_update_file,
            #                         headers=headers)
            
            response = requests.post("https://newsvibe.online/fresh_content",
                                     data=content_update_file,
                                     headers=headers)
        
            print(response)
        
        pass
    
    
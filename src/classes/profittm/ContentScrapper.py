
import numpy as np
import pandas as pd

import time
from datetime import datetime


from tqdm import tqdm
from pprint import pprint
from joblib import Parallel, delayed
from copy import deepcopy
from hashlib import sha256
import pandas as pd

import newspaper
from newspaper import Article, Source, Config
import nltk
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


class ContentScrapper():
    def __init__(self):
        
        pass
    
    def get_default_urls(self):
        
        """hot_requests = newspaper.hot()
        print(hot_requests)
        print(len(hot_requests))
        
        popular_urls = newspaper.popular_urls()
        pprint(popular_urls)
        print(len(popular_urls))
        
        newspaper.languages()"""
        
        popular_urls = newspaper.popular_urls()
        return popular_urls
    
    def get_custom_urls(self, urls_csv_path):
        urls = pd.read_csv( urls_csv_path )
        urls = urls["url"].to_numpy()
        
        return urls
        
    
    def set_urls_to_ignore(self, url_list):
        pass
    
    def postprocess_parsed_data(self, parsed_data, n_jobs=10):
        
        parsed_data = deepcopy( parsed_data )
    
        parsed_data = self.merge_data_( parsed_data )
        parsed_data, trash_urls = self.remove_duplicates_( parsed_data )
        parsed_data = self.select_english_language_content_( parsed_data, n_jobs )
        
        parsed_data = np.array( parsed_data )
        parsed_data = pd.DataFrame( parsed_data, columns=["url", "news_source", "content", "article_datetime", "parse_datetime"] )
        
        return parsed_data, trash_urls
    
    def merge_data_(self, parsed_data):
        
        dense_filled_data = []
        for i in range( len(parsed_data) ):
            articles_batch = parsed_data[i]
            if len(articles_batch) == 0:
                continue
            dense_filled_data.append( parsed_data[i] )
        
        merged_data = []
        for i in range( len(dense_filled_data) ):
            for j in range( len(dense_filled_data[i]) ):
                current_row = dense_filled_data[i][j]
                merged_data.append( current_row )
        return merged_data
    
    def remove_duplicates_(self, parsed_data):
        
        
        content_hash_counts = {}
        for i in range( len(parsed_data) ):
            current_row = parsed_data[i]
            text_content = str(current_row[2]).encode("utf-8") # for None case
            text_hash = sha256(text_content, usedforsecurity=True)
            text_hash = text_hash.hexdigest()
            if text_hash in content_hash_counts.keys():
                content_hash_counts[text_hash] += 1
            else:
                content_hash_counts[text_hash] = 1
        
        clean_data = []
        trash_urls = []
        for i in range( len(parsed_data) ):
            current_row = parsed_data[i]
            text_content = current_row[2]
            text_content = str(current_row[2]).encode("utf-8") # for None case
            text_hash = sha256(text_content, usedforsecurity=True)
            text_hash = text_hash.hexdigest()
            
            if content_hash_counts[text_hash] == 1:
                clean_data.append( current_row )
            else:
                trash_urls.append( current_row[0] )
            
        return clean_data, trash_urls
    
    def select_english_language_content_(self, parsed_data, n_jobs=10):
        
        def process_batch(parsed_data_batch):
        
            def get_lang_detector(nlp, name):
                return LanguageDetector()
            
            #spacy.cli.download("en")
            nlp = spacy.load("en_core_web_sm")
            Language.factory("language_detector", func=get_lang_detector)
            nlp.add_pipe('language_detector', last=True)
            
            english_content = []
            for i in tqdm(range( len(parsed_data_batch) ), desc="Selecting english language content"):
                current_row = parsed_data_batch[i]
                text_content = current_row[2]
                
                text_content = nlp(text_content)
                text_language = text_content._.language["language"]
                text_language_score = text_content._.language["score"]
                
                if (text_language == "en") and (text_language_score >= 0.95):
                    english_content.append( current_row )
            return english_content
            
        
        data_batches = np.array_split( parsed_data, n_jobs )
        content_batches = Parallel(n_jobs=n_jobs)(delayed(process_batch)(data_batch) for data_batch in data_batches )
        
        filled_batches = []
        filled_batches_ids = []
        for i in range(len(content_batches)):
            if len(content_batches[i]) > 0:
                filled_batches.append(content_batches[i])
                filled_batches_ids.append(i)
        #filled_batches = np.array( filled_batches )
        #content_batches = filled_batches
        english_content = np.vstack( filled_batches )
        print(english_content.shape)
        
        return english_content
    
    def parse_fresh_data(self, url_list, sleep_time=0.2, verbose=True, n_jobs=10, urls_to_ignore=None, proxies=None):
        
        fresh_parsed_data = self.parse_site_list(url_list=url_list, sleep_time=sleep_time, verbose=verbose, n_jobs=n_jobs, urls_to_ignore=urls_to_ignore, proxies=proxies)
        fresh_parsed_data, trash_urls = self.postprocess_parsed_data( fresh_parsed_data, n_jobs=n_jobs )
        
        return fresh_parsed_data, trash_urls
        
    
    def parse_site_list(self, url_list, sleep_time=0.2, verbose=True, n_jobs=1, urls_to_ignore=None, proxies=None):
        
        start_time = datetime.now()
        
        urls_to_ignore = set(urls_to_ignore)
        
        parsed_data = []
        for i in tqdm(range(len(url_list)), desc="Parsing news sites"):
            current_url = url_list[i]
            parse_result = self.parse_current_site(current_url, sleep_time=sleep_time, verbose=verbose, n_jobs=n_jobs, urls_to_ignore=urls_to_ignore, proxies=proxies)
            parsed_data.append( parse_result )
        
        total_time = datetime.now() - start_time
        print("Total parsing time: {}".format( total_time ))
        
        return parsed_data
    
    def parse_current_site(self, site_url, sleep_time=0.1, verbose=False, n_jobs=1, urls_to_ignore=None, proxies=None):
        
        config = Config()
        config.memoize_articles = False
        config.fetch_images = False
        config.number_threads = 1
        config.request_timeout = 10
        config.verbose = True
        #config.MIN_WORD_COUNT = 80
        #config.MIN_SENT_COUNT = 4
        
        if proxies is not None:
            config.proxies = proxies
        
        if verbose:
            print("Scrapping articles on {}".format(site_url))
        article_source = Source( site_url, config=config )
        article_source.build()
            
        article_urls = []
        for i in range(article_source.size()):
            current_article_url = article_source.articles[i].url
            if current_article_url in urls_to_ignore:
                continue
            article_urls.append( current_article_url )
        
        if verbose:
            print("Article urls to parse count: {}".format(len(article_urls)))
            print("All: {} | Ignored: {} | New: {}".format(article_source.size(), article_source.size() - len(article_urls), len(article_urls)))
    
        def parse_article(article_url, sleep_time, config):
            article = Article( article_url, keep_article_html=True, config=config )
            try:
                article.download()
                article.parse()
            except Exception as e:
                print(e)
                return [article_url, None, None, None, str(datetime.now())]
                
            article_title = str(article.title)
            article_text = str(article.text)
            article_publish_date = None #article.publish_date
            
            if len(article_text.split()) == 0:
                return [article_url, None, None, None, str(datetime.now())]
            
            time.sleep(sleep_time)
            
            parsed_article = [article_url, article_title, article_text, article_publish_date, str(datetime.now())]
            
            return parsed_article
        
        parsed_articles = Parallel(n_jobs=n_jobs, verbose=10)(delayed(parse_article)(url, sleep_time=sleep_time, config=config) for url in article_urls)
        
        return parsed_articles
    
    
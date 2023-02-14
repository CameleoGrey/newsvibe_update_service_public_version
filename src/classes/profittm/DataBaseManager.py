
import pandas as pd
from copy import deepcopy
from src.classes.paths_config import *

import sqlite3

class DataBaseManager():
    def __init__(self, db_path):
        
        self.db_connect = sqlite3.connect( db_path )
        self.cursor = self.db_connect.cursor()
        
        self.check_tables_()
        
        pass
    
    def check_tables_(self):
        
        check_parsed_data_exists = """
        CREATE TABLE IF NOT EXISTS parsed_data (
            
        id INTEGER PRIMARY KEY,
        url VARCHAR NOT NULL,
        news_source VARCHAR,
        content LONGTEXT,
        article_datetime VARCHAR,
        parse_datetime VARCHAR
        );
        """
        self.cursor.execute( check_parsed_data_exists )
        
        check_trash_urls_exists = """
        CREATE TABLE IF NOT EXISTS trash_urls (
        id INTEGER PRIMARY KEY,
        url VARCHAR NOT NULL
        );
        """
        self.cursor.execute( check_trash_urls_exists )
        
        self.db_connect.commit()
        
        pass
    
    def get_parsed_data(self):
        
        get_data_query = """
        SELECT * FROM parsed_data;
        """
        
        selected_data = self.cursor.execute( get_data_query )
        gathered_data = selected_data.fetchall()
        for i in range(len(gathered_data)):
            gathered_data[i] = list(gathered_data[i])[1:]
        
        columns = ["url", "news_source", "content", "article_datetime", "parse_datetime"]
        if len(gathered_data) == 0:
            existed_data_df = pd.DataFrame(columns=columns)
        else:
            existed_data_df = pd.DataFrame( data=gathered_data, columns=columns )
        
        return existed_data_df
    
    def get_trash_only_urls(self):
        
        get_trash_urls_query = """
        SELECT url FROM trash_urls;
        """
        
        selected_data = self.cursor.execute( get_trash_urls_query )
        gathered_data = selected_data.fetchall()
        for i in range(len(gathered_data)):
            gathered_data[i] = gathered_data[i][0]
        
        return gathered_data
    
    def get_urls_to_ignore(self):
        get_already_parsed_urls = """
        SELECT url FROM parsed_data;
        """
        selected_data = self.cursor.execute( get_already_parsed_urls )
        already_parsed_urls = selected_data.fetchall()
        for i in range(len(already_parsed_urls)):
            already_parsed_urls[i] = already_parsed_urls[i][0]
            
        trash_urls = self.get_trash_only_urls()
        
        urls_to_ignore = already_parsed_urls + trash_urls
        
        return urls_to_ignore
    
    def update_parsed_data(self, fresh_data):
        
        data_to_insert = fresh_data.to_numpy()
        
        for i in range( len(data_to_insert) ):
            self.cursor.execute( "INSERT INTO parsed_data (url, news_source, content, article_datetime, parse_datetime) VALUES(?,?,?,?,?);", tuple(data_to_insert[i]) )
            
        self.db_connect.commit()
        
        pass
    
    def update_trash_only_urls(self, fresh_trash_urls):
        
        fresh_trash_urls = deepcopy( fresh_trash_urls )
        #fresh_trash_urls = [ (fresh_trash_urls[i], ) for i in range(len(fresh_trash_urls)) ]
        
        for i in range( len(fresh_trash_urls) ):
            self.cursor.execute( "INSERT INTO trash_urls (url) VALUES(?);", (fresh_trash_urls[i],) )
        self.db_connect.commit()
        
        pass
    
    def delete_all_parsed_data_(self):
        
        drop_parsed_data = """
        DROP TABLE IF EXISTS parsed_data;
        """
        self.cursor.execute( drop_parsed_data )
        
        check_parsed_data_exists = """
        CREATE TABLE IF NOT EXISTS parsed_data (
            
        id INTEGER PRIMARY KEY,
        url VARCHAR NOT NULL,
        news_source VARCHAR,
        content LONGTEXT,
        article_datetime VARCHAR,
        parse_datetime VARCHAR
        );
        """
        self.cursor.execute( check_parsed_data_exists )
        self.db_connect.commit()
        
        pass
    
    def delete_trash_urls_(self):
        
        drop_trash_urls = """
        DROP TABLE IF EXISTS trash_urls;
        """
        self.cursor.execute( drop_trash_urls )
        
        check_trash_urls_exists = """
        CREATE TABLE IF NOT EXISTS trash_urls (
        id INTEGER PRIMARY KEY,
        url VARCHAR NOT NULL
        );
        """
        self.cursor.execute( check_trash_urls_exists )
        
        self.db_connect.commit()
        
        pass
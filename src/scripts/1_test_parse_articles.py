
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.classes.utils import *
from src.classes.paths_config import *
from src.classes.profittm.ContentScrapper import ContentScrapper
from src.classes.profittm.DataBaseManager import DataBaseManager


if __name__ == "__main__":
    
    database_manager = DataBaseManager( database_path )
    content_scrapper = ContentScrapper()
    #urls_for_parsing = content_scrapper.get_default_urls()
    urls_for_parsing = content_scrapper.get_custom_urls( Path(interim_dir, "valid_source_urls.csv") )
    urls_to_ignore = database_manager.get_urls_to_ignore()
    
    ##########
    #debug
    #urls_for_parsing = urls_for_parsing[1:]
    ##########
    
    #parsed_data = content_scrapper.parse_site_list(urls_for_parsing, sleep_time=0.2, verbose=True, n_jobs=10, urls_to_ignore=urls_to_ignore)
    #postprocessed_data = content_scrapper.postprocess_parsed_data( parsed_data, n_jobs=10 )
    #postprocessed_data.to_parquet( Path(interim_dir, "parsed_data.parquet"), index=False )
    
    proxies = {"http": "http://proxy_pool_login:password@proxy_pool_ip:port"}
    fresh_parsed_data, fresh_trash_urls = content_scrapper.parse_fresh_data(url_list=urls_for_parsing, sleep_time=0.2, verbose=True, n_jobs=10, urls_to_ignore=urls_to_ignore, proxies=proxies)
    database_manager.update_parsed_data( fresh_parsed_data )
    database_manager.update_trash_only_urls( fresh_trash_urls )
    
    print("done")
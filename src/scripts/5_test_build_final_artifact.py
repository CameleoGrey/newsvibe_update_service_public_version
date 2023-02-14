

import pandas as pd
from pprint import pprint
from datetime import datetime
from time import gmtime, strftime
import json

import numpy as np
import os

import matplotlib.pyplot as plt
from src.classes.profittm.ContentScrapper import ContentScrapper
from src.classes.profittm.TopicInterpreter import TopicInterpreter
from src.classes.profittm.DataBaseManager import DataBaseManager
from src.classes.utils import load, save
from src.classes.paths_config import *

content_scrapper = ContentScrapper()
urls_for_parsing = content_scrapper.get_custom_urls( Path(interim_dir, "valid_source_urls.csv") )
sources_count = len( urls_for_parsing )

database_manager = DataBaseManager( database_path )
parsed_data = database_manager.get_parsed_data()
parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
parsed_data = parsed_data[ parsed_data["parse_datetime"] >= datetime(day=14, month=12, year=2022) ]
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

data_for_frontend = {}
data_for_frontend["article_summaries"] = list(article_summaries)
data_for_frontend["topic_summaries"] = group_summaries
data_for_frontend["news_count"] = len(article_summaries[0])
data_for_frontend["sources_count"] = sources_count
data_for_frontend["update_time"] = str(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

with open(Path(interim_dir, "data_for_frontend.json"), "w") as data_for_frontend_file:
    json.dump(data_for_frontend, data_for_frontend_file)

save( data_for_frontend, Path(interim_dir, "data_for_frontend.pkl") )



print("done")
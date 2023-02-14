
from pathlib import Path
from src.classes.utils import *
from src.classes.paths_config import *

from src.classes.profittm.UpdateContentPipeline import UpdateContentPipeline

update_pipeline = UpdateContentPipeline()
save( update_pipeline, Path(interim_dir, "update_pipeline.pkl") ) # just to save update time
proxies = {"http": "http://proxy_pool_login:password@proxy_pool_ip:port"}
update_pipeline.update_parsed_data(proxies=proxies, use_default_urls=False)
update_pipeline.retrain_topic_model(fit_vectorizer_on_fresh_texts=True)
update_pipeline.make_fresh_summaries()
update_pipeline.plot_useful_graphics()
update_pipeline.build_fresh_frontend_content()
update_pipeline.send_update()

print("done")


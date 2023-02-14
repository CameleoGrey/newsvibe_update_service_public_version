
import os
import os
import sys
import pathlib
from pathlib import Path
project_dir = os.path.abspath(os.getcwd())
project_dir = "\\".join( project_dir.split("\\")[:-2] )
sys.path.append( project_dir )

models_dir = os.path.join(project_dir, "models")
if not Path( models_dir ).exists():
    Path( models_dir ).mkdir(parents=True, exist_ok=True)

data_dir = os.path.join(project_dir, "data")
if not Path( data_dir ).exists():
    Path( data_dir ).mkdir(parents=True, exist_ok=True)

interim_dir = os.path.join( data_dir, "interim" )
if not Path( interim_dir ).exists():
    Path( interim_dir ).mkdir(parents=True, exist_ok=True)

raw_dir = os.path.join( data_dir, "raw" )
if not Path( raw_dir ).exists():
    Path( raw_dir ).mkdir(parents=True, exist_ok=True)

images_dir = os.path.join( data_dir, "images" )
if not Path( images_dir ).exists():
    Path( images_dir ).mkdir(parents=True, exist_ok=True)

production_dir = os.path.join( data_dir, "production" )
if not Path( production_dir ).exists():
    Path( production_dir ).mkdir(parents=True, exist_ok=True)

database_path = os.path.join( production_dir, "newsvibe.db" )

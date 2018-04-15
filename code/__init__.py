#standard tools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep
import pymongo
import requests

# local libraries
from lib.wiki_api import WikiAPI , MyWikiDB , WikiSearch

#for semantic search
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup

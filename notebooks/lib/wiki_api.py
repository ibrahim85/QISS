import requests
from bs4 import BeautifulSoup
import pandas as pd
import re 
import spacy
import nltk
from nltk.corpus import stopwords
import time
import pymongo
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# MyWikiDB - created database connection

class MyWikiDB():
    
    def __init__(self):
        self.client=pymongo.MongoClient('localhost', 27017)
        self.dbr=self.client.myWiki

    def check_data_loaded(self):
        ''' output dataframe of categories and pages loaded'''
        
        loads_cllr=self.dbr.loads_collection
        cursor=loads_cllr.find()
        cdocs=pd.DataFrame(list(cursor))
        try:
            cdocs=cdocs.drop('_id',1)
        except:
            cdocs        
        display(cdocs)
        
        pageload_cllr=self.dbr.pageload_collection
        cursor=pageload_cllr.find()
        pdocs=pd.DataFrame(list(cursor))
        try:
            pdocs=pdocs.drop('_id',1)
        except:
            pdocs        
        display(pdocs)
        
        return 

# WikiSearch - searchs loaded pages for related articles
        
class WikiSearch():
    
    '''
    DTM created when class instantiated
    functions:
    
    __init__ -> calls build_dtm_matrix -> calls build_corpus ->
    returns dtm sparse matrix , dtm index and fitted tf-idf
    
    search_myWiki -> returns  top 10 related articles
    
    '''
    
    def __init__(self):
        # build dtm matrix each time class instantiated - can then run multiple searchs
        self.build_dtm_matrix()    
    
    def build_corpus(self,):
        ''' Load pages collection from mongodb and create dataframe for semantic analysis'''
        
        #load pages collection
        myWiki=MyWikiDB()
        pages_cllr=myWiki.dbr.page_collection
        cursor=pages_cllr.find()
        pages=list(cursor)
        
        #put in df, drop columns that are not required, set title as index
        corpus_df=pd.DataFrame(pages)
        columns_to_drop=['ns', 'category_list','pull_category', '_id']
        corpus_df.drop(columns_to_drop, axis=1, inplace=True)
        corpus_df.set_index('title', inplace=True)      
        self.corpus_df=corpus_df
        return 
        
    def build_dtm_matrix(self,n_components=100):
        ''' 
        Build document term matrix and svd matrix for page extracts held in the mongodb pages_collection
        USE sparse matrix rather than dfs as cause memory issues
        RE-RUN each time new content added to pages_collection
        ''' 
        self.build_corpus()
        
        tfidf_v = TfidfVectorizer(min_df = 1, stop_words = 'english')
        dtm_sp = tfidf_v.fit_transform(self.corpus_df.extract)
        dtm_sp_index=list(self.corpus_df.index)
        self.tfidf_v=tfidf_v
        self.dtm_sp=dtm_sp
        self.dtm_sp_index=dtm_sp_index  
        return 
        
    def search_myWiki(self,search_term ):
        
        s_term=[search_term]
        
        if search_term=="":
            return 'Please input search term'
        
        else :
            #vectorize search term with TFIDF vectorizer (previously fitted)
            s_term_encoded=self.tfidf_v.transform(s_term)
            print('step 1 done')
            
            #add to existing document term matrix
            dtma_sp=self.dtm_sp.copy()
            dtma_sp.data=np.append(self.dtm_sp.data,s_term_encoded.data)
            dtma_sp.indices=np.append(self.dtm_sp.indices,s_term_encoded.indices)
            dtma_sp.indptr=np.append(self.dtm_sp.indptr,(self.dtm_sp.indptr[-1]+s_term_encoded.indptr[-1]))
            dtma_sp._shape=(self.dtm_sp.shape[0]+1 , self.dtm_sp.shape[1])
            print('step 2 done')
            
            #re fit SVD to augmented document term matrix
            svd_v = TruncatedSVD(n_components=100)
            component_names = ["component_"+str(i+1) for i in range(svd_v.n_components)]
            svd_index=self.dtm_sp_index.copy()
            svd_index.append('search_term='+search_term)
            svdma=svd_v.fit_transform(dtma_sp)
            svdma_df=pd.DataFrame(svdma,index=svd_index , columns=component_names)
            print('step 3 done')
            
            #find index of search term in refit SVD matrix
            s_term_svd_vector=svdma_df.loc['search_term='+search_term,:].values.reshape(1,-1)
            print('step 4 done')
            
            #calculate cosine similarty of search term against other SVD vectors
            svdma_df['cosine_sim'] = cosine_similarity(svdma_df, s_term_svd_vector)
            print('step 5 done')
            
            return svdma_df[['cosine_sim']].sort_values('cosine_sim', ascending=False).head(10)

# WikiAPI - loads requested category 
        
class WikiAPI():

    """
    functions:
    'wiki_cats'  -> calls 'pull_wiki_data' -> calls 'read_categories' -> 
    returns unique list of subcategories & ids,
    writes subcategories to self.subcategories ready to load
    
    'write_subcats_to_mongo' -> writes contents of self.subcats to 
    category collection in mongodb
    
    'wiki_pages' -> calls 'pull_wiki_data' - > calls 'read_pages' -> 
    returns unique no. pages & no. duplicates,
    results written to self.pages ready for 'load_articles'
    
    'load_articles' -> calls 'read_articles' -> calls 'pull_wiki_page' & 'cleaner'
    'load_articles_test' - loads to pagetest collection  - version for testing during development
    """    
    
    def __init__(self, cat_name,depth=2, run=True):
        self.cat_name=cat_name
        self.depth=depth
        self.pages=[]
        self.subcats=[]
        nltk.download('stopwords')
        self.nlp=spacy.load('en')
        self.nltk_stop=stopwords.words('english')
        self.nltk_stop.append('displaystyle')
        self.check=''
        if run==True:
            self.wiki_cats()
            print(self.check)
            if self.check!='stop':
                self.write_subcats_to_mongo()
                self.wiki_pages()
                load_pages=input('Continue with load (y / n) ? :')
                if load_pages=='y':
                      self.load_articles()
                else:
                      print('load aborted')
       
        return  
         
    def wiki_cats(self, cat_name='head', subcats_ids=set(), r=0, depth=2):
        ''' 
        Recursive function to pull subcategories from the 
        whole category tree and remove duplicates
        depth controls the no of levels of subcategories that are picked up
        '''
        
        if cat_name=='head':
            cat_name=self.cat_name
            depth=self.depth
            try:
                subcats_r = self.read_categories(self.pull_wiki_data(cat_name))
            except:
                print('category does not exist, please try again')
                self.check='stop'
                return
                
        #temporary holders for sub-category collection
        s_subcats=[]
        
        subcats_r = self.read_categories(self.pull_wiki_data(cat_name))
        subcats=[s for s in subcats_r if s['pageid'] not in subcats_ids]
        
        for i in subcats:
            if i['pageid'] not in subcats_ids:
            
                #check subcat not already added
                subcats_ids.add(i['pageid'])
                
                r+=1
                if r<(depth+1):
                    s_subcats, subcats_ids= self.wiki_cats(i['title'],subcats_ids, r, depth)
                    #add new subcats not already in subcats
                    new_s_subcats=[s for s in s_subcats if s not in subcats]
                    subcats=subcats+new_s_subcats
                r-=1
        
        self.subcats=subcats      
        return subcats , subcats_ids 
    
    def wiki_pages(self):
        '''
        function to pull pages for 
        unique subcategories , avoiding duplicates
        capture first category from which is page is pulled
        for data checking
        '''
        page_ids=set()

        pages = self.read_pages(self.pull_wiki_data(self.cat_name))
        for p in pages:
            page_ids.add(p['pageid'])
            p['pull_category']=self.cat_name

        for subcat in self.subcats:

            pages_r = self.read_pages(self.pull_wiki_data(subcat['title']))

            for pg in pages_r:
                if pg['pageid'] not in page_ids:
                    pg['pull_category']=subcat['title']
                    page_ids.add(pg['pageid'])
                    pages.append(pg)
        
        self.pages=pages
        to_add=len(pages)
        time_to_add=to_add*.5/60
        
        return print('Unique pages to be added {} , approx load time (.5 seconds per page) {:.2}mins'.format(to_add, time_to_add))
  

    def write_subcats_to_mongo(self):
        '''
        write subcategory info to Mongo categories_collections
        in case want it later
        '''
        #first add _id = pageid as mongo primary key 
        for subcat in self.subcats:
            subcat['_id']=subcat['pageid']
        
        #insert subcats
        myWiki=MyWikiDB()
        subcat_cllr=myWiki.dbr.category_collection
        start=subcat_cllr.find().count()
        d=0
        try:
            subcat_cllr.insert_many(self.subcats, ordered=False)
        except:
            print('no subcategories loaded') 
                
        end=subcat_cllr.find().count()
        added=end - start
        dropped=len(self.subcats) - added
        
        #record master categories loaded plus date in mongoDB
        cat_loaded={'master_cat': self.cat_name , 'sub_cats_added':(end-start), 'loaded':time.strftime("%c")}
        cats_loaded_cllr=myWiki.dbr.loads_collection
        cats_loaded_cllr.insert_one(cat_loaded)
            
        return print('subcategories added {}, duplicates not loaded {}'. format(added, dropped))
 
    def load_articles_test(self):
        ''' 
        TEST version - used for testing load process and semantic search developement
        for each page add article info and load to Mongo page collection
        collections primary key = pageid  , if page exists it will not be added
        '''
        d=0
        myWiki=MyWikiDB()
        pagetest_cllr=myWiki.dbr.pagetest_collection
        start=pagetest_cllr.find().count()
        i=0
        for page in self.pages:
            i+=1
            print('loading page {}, {}'.format(i, page['title']))
            try:
                pagetest_cllr.insert_one(self.read_articles(page))
                print('loaded',page)
            except:
                print('not loaded',page)
                d+=1
        end=pagetest_cllr.find().count()   
 
        #record pages loaded plus date in mongoDB
        pages_loaded={'master_cat': self.cat_name , 'pages_added':(end-start), 'loaded':time.strftime("%c")}
        pages_loaded_cllr=myWiki.dbr.pagetestload_collection
        pages_loaded_cllr.insert_one(pages_loaded)

        return print('pages added {}, duplicates droped {}'. format(end-start,d)) 

    def load_articles(self):
        ''' 
        for each page add article info and load to Mongo page collection
        collections primary key = pageid  , if page exists it will not be added
        '''
        d=0
        myWiki=MyWikiDB()
        page_cllr=myWiki.dbr.page_collection
        start=page_cllr.find().count()
        i=0
        for page in self.pages:
            i+=1
            print('loading page {}, {}'.format(i, page['title']))
            try:
                page_cllr.insert_one(self.read_articles(page))
            except:
                d+=1
        end=page_cllr.find().count()
        
        #record pages loaded plus date in mongoDB
        pages_loaded={'master_cat': self.cat_name , 'pages_added':(end-start), 'loaded':time.strftime("%c")}
        pages_loaded_cllr=myWiki.dbr.pageload_collection
        pages_loaded_cllr.insert_one(pages_loaded)
 
        return print('pages added {}, duplicates droped {}'. format(end-start,d))
        
    
    def pull_wiki_data(self, cat_name):
        '''
        Query page and subcategory for a specified category using wikipedia API
        '''
        
        params = {"action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": cat_name ,
        "cmlimit": "max"
        }
       
        response = requests.get("https://en.wikipedia.org/w/api.php", params = params)
        data=response.json()
        
        return data
    
    def pull_wiki_page(self, page):
        '''
        pull page extract and category list for a specified page using 
        wikipedia API
        '''
        
        #pull page extract
        params_p = {"action": "query",
        "titles": page,
        "prop": 'extracts' ,
        "format": 'json' ,
        }
        
        response_p = requests.get("https://en.wikipedia.org/w/api.php", params = params_p)
        data = response_p.json()
        
        #pull page categories
        params_c = {"action": "query",
        "titles": page,
        "prop": 'categories' ,
        "format": 'json' ,
        }
        
        response_c = requests.get("https://en.wikipedia.org/w/api.php", params = params_c)
        cats = response_c.json()
        
        return data , cats
 

    def read_pages(self,data):
        ''' read all pages for category from wikipedia API query'''
        return list(filter(lambda x: x["ns" ] == 0, data["query"]["categorymembers"]))

    
    def read_categories(self,data):
        ''' read all subcategories for category from wikipedia API query'''
        return list(filter(lambda x: x["ns"] == 14, data["query"]["categorymembers"]))
    
    
    def read_articles(self, page):
        '''
        read page extract and category list from wikipedia 
        API query and add to page dict
        '''
        data, cats=self.pull_wiki_page(page['title'])
  
        # first data clean - extract from json query and parse html
        article=data['query']['pages'][str(page['pageid'])]['extract']
        soup = BeautifulSoup(article, 'html.parser')
        extract=soup.get_text()
 
        # 2nd data clean including lemmatization and stop words
        extract_clean=self.cleaner(extract)
        
        cats_p=[c['title'] for c in cats['query']['pages'][str(page['pageid'])]['categories']]

        page['category_list']=cats_p
        page['extract']=extract_clean
        
        #Add _id field as pageid - will be used as a unique index by Mongo
        page['_id']=page['pageid']
        #print('5 - page',page)
        return page   
    
    def cleaner_old(self,text):
        ''' Clean text data, apply spacy lemmatization and nltk stop words'''
        text = re.sub('<* />','',text)
        text = re.sub('<.*>.*</.*>','', text)
        text = re.sub('[\d]',' ',text)
        text = re.sub('{*}',' ',text)
        text = re.sub('[\n]',' ',text)
        text = re.sub('[^a-zA-Z\s]',' ',text)
        text = ' '.join(i.lemma_ for i in self.nlp(text)
                    if i.orth_ not in self.nltk_stop)
        text = ' '.join(text.split())
        return text
    
    def cleaner(self,text):
        ''' Clean text data, apply spacy lemmatization and nltk stop words'''
        text = re.sub('{.*}',' ',text)
        text = re.sub('[^a-zA-Z ]',' ',text) # remove numbers and characters not in latin alphabet 
        text = ' '.join(i.lemma_ for i in self.nlp(text)
                    if i.lemma_ not in self.nltk_stop)
        text = re.sub('-PRON-',' ',text)  # added by spacy lemmatization ?? - remove
        text = ' '.join(i for i in text.split() if len(i)!=1)  # remove redundant spaces and individual letters
`       return text

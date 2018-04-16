import json
import urllib
import urllib.request
import urllib.parse
import re
import spacy
import nltk
from nltk.corpus import stopwords
from enum import Enum


class Content:
    id = None
    parent = None
    title = None
    children = None


class InformationType(Enum):
    SubCategory = 'subcat'
    Article = 'page'


def download_content(content_type, content_name):
    form_data = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': content_name,
        'format': 'json',
        'cmlimit': '500',
        'cmtype': content_type
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(form_data)
    endpoint = urllib.request.urlopen(url)
    request_content = endpoint.read()
    encoding = endpoint.info().get_content_charset('utf-8')
    json_content = json.loads(request_content.decode(encoding))
    result = json_content['query']['categorymembers']
    resultset = list()
    for r in result:
        page_content = Content()
        page_content.parent = content_name
        page_content.id = r['pageid']
        page_content.title = r['title']
        resultset.append(page_content)
    return resultset


def download_collection_intro(page_ids):
    form_data = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'exlimit': 'max',
        'pageids': "|".join(page_ids),
        'redirects': '1'
    }

    url = "https://en.wikipedia.org/w/api.php?"+ urllib.parse.urlencode(form_data) + "&explaintext&exintro"
    endpoint = urllib.request.urlopen(url)
    request_content = endpoint.read()
    encoding = endpoint.info().get_content_charset('utf-8')
    json_content = json.loads(request_content.decode(encoding))
    for page_id in page_ids:
        # for now, skipping redirects
        if page_id in json_content['query']['pages']:
            summary = json_content['query']['pages'][page_id]['extract']
            write_content("output/pages/" + page_id + ".txt", summary, True)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def write_content(file_path, file_content, clean_content=False):
    if clean_content:
        file_content = cleaner(file_content)
    fn = file_path
    f = open(fn, 'w')
    f.write(file_content)
    f.flush()
    f.close()


def cleaner(text_content):
    ''' Clean text data, apply spacy lemmatization and nltk stop words'''
    text_content = re.sub("\.+\s",'\n', text_content)
    sentences = list()
    for text in text_content.split('\n'):
        text = re.sub('{.*}', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)  # remove numbers and characters not in latin alphabet
        text = ' '.join(i.lemma_ for i in words(text)
                        if i.lemma_ not in stop_words)
        text = re.sub('-PRON-', ' ', text)  # added by spacy lemmatization ?? - remove
        text = ' '.join(i for i in text.split() if len(i) != 1)  # remove redundant spaces and individual letters
        sentences.append(text)

    return "\n".join(sentences)

nltk.download('stopwords')
words = spacy.load('en')
stop_words = stopwords.words('english')
if __name__ == '__main__':

    categories_to_search = ['Category:Machine learning', 'Category:Business software']
    content_to_scrape = dict()
    content_to_scrape[InformationType.SubCategory] = categories_to_search  # reusable when scraping specific pages
    scraped_content = dict()
    for content_key in content_to_scrape.keys():
        for content_val in content_to_scrape[content_key]:
            scraped_content[content_val] = download_content(content_key.value, content_val)

    categories = list()
    pages = list()
    separator = "###"
    max_number_of_extracts = 20  # wikimedia api limitation
    for scraped_key in scraped_content.keys():
        list_of_contents = scraped_content[scraped_key]
        for content in list_of_contents:
            categories.append("{0}{1}{2}{1}{3}".format(scraped_key, separator, content.id, content.title))
            content.children = download_content(InformationType.Article.value, content.title)
            page_ids = []
            for children in content.children:
                pages.append("{0}{1}{2}{1}{3}{1}{4}{1}{5}".format(
                    scraped_key, separator, content.id, content.title, children.id, children.title))
                page_ids.append(str(children.id))
            batched_ids = batch(page_ids, max_number_of_extracts)
            for b in batched_ids:
                download_collection_intro(b)

    write_content("output/categories.txt", "\n".join(categories))
    write_content("output/pages.txt", "\n".join(pages))

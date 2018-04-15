import json
import urllib
import urllib.request
import urllib.parse
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


def write_content(file_path, file_content):
    fn = file_path
    f = open(fn, 'w')
    f.write(file_content)
    f.flush()
    f.close()


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
    for scraped_key in scraped_content.keys():
        list_of_contents = scraped_content[scraped_key]
        for content in list_of_contents:
            categories.append("{0}{1}{2}{1}{3}".format(scraped_key, separator, content.id, content.title))
            content.children = download_content(InformationType.Article.value, content.title)
            for children in content.children:
                pages.append("{0}{1}{2}{1}{3}{1}{4}{1}{5}".format(
                    scraped_key, separator, content.id, content.title, children.id, children.title))

    write_content("output/categories.txt", "\n".join(categories))
    write_content("output/pages.txt", "\n".join(pages))

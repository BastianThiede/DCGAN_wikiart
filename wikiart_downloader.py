from pprint import pprint
import json
import requests
from bs4 import BeautifulSoup
from json.decoder import JSONDecodeError

BASE_URL = 'https://www.wikiart.org'


def crawl_full():
    genre_urls = get_all_genres()
    for i,genre in enumerate(genre_urls):
        print('Parsing genre: {} {}/{}'.format(genre,i,len(genre_urls)))
        img_urls_genre = iter_through_genre(genre)
        print('Found {} links!'.format(len(img_urls_genre)))
        with open('url_links_wikiart_full.txt', 'a') as f:
            for img_url in img_urls_genre:
                f.write('{}\n'.format(img_url))


def parse_response(resp_json):
    image_urls = list()
    for item in resp_json['Paintings']:
        image_urls.append(item.get('image'))
    return image_urls


def iter_through_genre(genre_url):
    has_next = True
    page_count = 1
    genre_image_urls = list()
    while has_next:
        resp = get_page_response(genre_url, page_count)
        if resp and resp.get('Paintings'):
            img_urls = parse_response(resp)
            genre_image_urls.extend(img_urls)
            page_count += 1
        else:
            has_next = False

    return genre_image_urls


def get_page_response(genre_url, page_num):
    params = (
        ('json', '2'),
        ('layout', 'new'),
        ('page', page_num),
        ('resultType', 'masonry'),
    )
    response = requests.get(genre_url, params=params)
    try:
        retval = json.loads(response.text)
    except JSONDecodeError:
        return None

    return retval


def get_all_genres():
    response = requests.get('https://www.wikiart.org/en/paintings-by-genre')
    soup = BeautifulSoup(response.text, 'lxml')
    genre_links = list()
    for ul_tag in soup.find_all('ul'):
        if ul_tag.get('class', ['nothing'])[0] == 'dictionaries-list':
            for a_tag in ul_tag.find_all('a'):
                full_url = BASE_URL + a_tag.get('href').split('?')[0]
                genre_links.append(full_url)
    return genre_links




if __name__ == '__main__':
    crawl_full()

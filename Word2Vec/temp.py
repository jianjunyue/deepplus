import urllib


def get_movie_url(movie_name):#根据电影名称，生成搜索结果的URL
  host_url = 'http://s.dydytt.net/plus/search.php?kwtype=0&keyword='
  movie_sign = urllib.parse.quote(movie_name.encode('GBK'))
  search_url = host_url + movie_sign
  return search_url

movieName="Toy Story (1995)"
url=get_movie_url(movieName)
print(url)
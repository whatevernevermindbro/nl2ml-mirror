import requests


def download_file(link, target_file, cookie_list=None, chunk_size=1024):
    cookies = dict(map(lambda cookie: (cookie["name"], cookie["value"]), cookie_list))
    resp = requests.get(link, stream=True, cookies=cookies)
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if chunk:
            target_file.write(chunk)

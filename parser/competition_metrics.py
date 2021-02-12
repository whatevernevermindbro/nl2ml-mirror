import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

COMPETITION_REFS_FILE = "competitions_ref.csv"
COMPETITIONS_FILE = "competitions.csv"
METRIC_NAMES_FILE = "metric_names.txt"

known_metrics = set(line.strip() for line in open(METRIC_NAMES_FILE))


def get_metric(url):
    url = "https://www.kaggle.com/c/" + url
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    res = soup.find_all('script')
    soup = BeautifulSoup(str(res[-1]), 'html.parser')
    for match in soup.findAll("script"):
        match.replaceWithChildren()
    line = str(soup)
    start = "Kaggle.State.push("
    line = line[line.find(start) + len(start):]
    line = line[:line.rfind("}") + 1]
    json_data = json.loads(line)
    for i in range(len(json_data["categories"]["tags"])):
        name = json_data["categories"]["tags"][i]["name"]
        if '@' in name:
            name = name[:name.find("@")]
        if name in known_metrics:
            return json_data["categories"]["tags"][i]["name"]
    return "unknown metric"


df = pd.read_csv(COMPETITION_REFS_FILE)
df['metric'] = df.apply(lambda row: get_metric(row.ref), axis=1)
df.to_csv(COMPETITIONS_FILE)

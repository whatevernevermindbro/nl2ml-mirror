import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
import csv

COMPETITION_REFS_FILE = "competitions_refs.csv"
COMPETITIONS_FILE = "competitions_info.csv"
METRIC_NAMES_FILE = "metric_names.txt"

known_metrics = set(line.strip() for line in open(METRIC_NAMES_FILE))


def get_tags(url):
    url = "https://www.kaggle.com/" + url
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    title = str(soup.find_all('title')[0])
    title = title.replace("<title>", "")
    title = title.replace(" | Kaggle</title>", "")

    res = soup.find_all('script')
    soup = BeautifulSoup(str(res[-1]), 'html.parser')

    for match in soup.findAll("script"):
        match.replaceWithChildren()
    line = str(soup)
    start = "Kaggle.State.push("
    line = line[line.find(start) + len(start):]
    line = line[:line.rfind("}") + 1]
    json_data = json.loads(line)
    metric = "unknown metric"
    data_type = ""
    subject = ""
    description = ""
    competition_type = json_data["hostSegment"]
    if json_data["overview"] is not None:
        description = json_data["overview"]["content"]
    technique = ""
    for i in range(len(json_data["categories"]["tags"])):
        name = json_data["categories"]["tags"][i]["name"]
        full_path = json_data["categories"]["tags"][i]["fullPath"]
        if '@' in name:
            name = name[:name.find("@")]
        if name in known_metrics:
            metric = name
        elif full_path is not None and "data type" in full_path:
            data_type = name
        elif full_path is not None and "subject" in full_path:
            subject = " ".join([subject, name])
        elif full_path is not None and "technique" in full_path:
            technique = name
    return [title, competition_type, description, metric, data_type, subject, technique]


df = pd.read_csv(COMPETITION_REFS_FILE)
data = np.array(list(df.apply(lambda row: get_tags(row["ref"]), axis=1)))

df[['comp_name', 'comp_type', 'Description', 'Metric', 'DataType', 'Subject', 'ProblemType']] = data
df.to_csv(COMPETITIONS_FILE, quoting=csv.QUOTE_NONNUMERIC, index=False)

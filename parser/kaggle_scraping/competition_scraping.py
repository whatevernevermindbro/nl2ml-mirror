import requests
import tempfile
import zipfile

import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# TODO: Get rid of xpath


def download_file(link, target_file, cookie_list=None, chunk_size=1024):
    cookies = dict(map(lambda cookie: (cookie["name"], cookie["value"]), cookie_list))
    resp = requests.get(link, stream=True, cookies=cookies)
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if chunk:
            target_file.write(chunk)


def _extract_public_leaderboard_score(leaderboard_row):
    score_block = leaderboard_row.find_element_by_xpath(
        "./td[contains(@data-th, 'Score')]"
    )
    return float(score_block.text)


def _load_full_leaderboard(webdriver):
    webdriver.log_in()

    leaderboard_data_link_container = webdriver.driver.find_element_by_xpath(
        "//div[@class='competition-leaderboard__actions']/a"
    )

    leaderboard_data_link = leaderboard_data_link_container.get_attribute("href")
    with tempfile.NamedTemporaryFile(suffix=".zip", buffering=0) as tmp_file:
        download_file(leaderboard_data_link, tmp_file, webdriver.driver.get_cookies())
        # this won't work on windows
        with zipfile.ZipFile(tmp_file.name, "r") as tmp_zip:
            csv_file = tmp_zip.namelist()[0]
            leaderboard_df = pd.read_csv(tmp_zip.open(csv_file))

    return leaderboard_df


def get_optimization_type(webdriver, kernel_link):
    webdriver.driver.get(kernel_link + "leaderboard")

    # Close cookie warning, if it is there
    webdriver.accept_cookies()

    leaderboard_rows = webdriver.driver.find_elements_by_css_selector(
        "tr.competition-leaderboard__row"
    )

    if len(leaderboard_rows) < 2:
        return None  # no way to determine if we should minimize or maximize

    best_score = _extract_public_leaderboard_score(leaderboard_rows[0])
    for i in range(len(leaderboard_rows) - 1):
        consecutive_places_diff = (
            _extract_public_leaderboard_score(leaderboard_rows[i]) -
            _extract_public_leaderboard_score(leaderboard_rows[i + 1])
        )
        if abs(consecutive_places_diff) > 1e-6:
            return "minimize" if consecutive_places_diff < 0 else "maximize"

    leaderboard_df = _load_full_leaderboard(webdriver)
    # For some annoying reason, leaderboard csv is sorted by team id and not by score
    if not (leaderboard_df["Score"] > best_score).any():
        return "maximize"
    elif not (leaderboard_df["Score"] < best_score).any():
        return "minimize"
    return None

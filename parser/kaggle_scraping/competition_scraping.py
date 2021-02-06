import requests
import tempfile
import zipfile

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from kaggle_scraping.base_scraper import BaseScraper


def download_file(link, target_file, cookie_list=None, chunk_size=1024):
    cookies = dict(map(lambda cookie: (cookie["name"], cookie["value"]), cookie_list))
    resp = requests.get(link, stream=True, cookies=cookies)
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if chunk:
            target_file.write(chunk)


class CompetitionScraper(BaseScraper):
    def __init__(self, max_load_wait=15.0):
        super().__init__(max_load_wait)

    @staticmethod
    def _extract_public_leaderboard_score(self, leaderboard_row):
        score_block = leaderboard_row.find_element_by_xpath(
            "./td[contains(@data-th, 'Score')]"
        )
        return float(score_block.text)

    def _load_full_leaderboard(self):
        self._log_in()

        leaderboard_data_link_container = self.driver.find_element_by_xpath(
            "//div[@class='competition-leaderboard__actions']/a"
        )

        leaderboard_data_link = leaderboard_data_link_container.get_attribute("href")
        with tempfile.NamedTemporaryFile(suffix=".zip", buffering=0) as tmp_file:
            download_file(leaderboard_data_link, tmp_file, self.driver.get_cookies())
            # this won't work on windows
            with zipfile.ZipFile(tmp_file.name, "r") as tmp_zip:
                csv_file = tmp_zip.namelist()[0]
                leaderboard_df = pd.read_csv(tmp_zip.open(csv_file))

        return leaderboard_df

    def get_optimization_type(self, kernel_link):
        self.driver.get(kernel_link + "leaderboard")

        # Close cookie warning, if it is there
        if not self.accepted_cookies:
            accept_cookies_button = self.driver.find_element_by_xpath(
                "//*[@id='site-container']/div/div[6]/div/div[2]/div"
            )
            accept_cookies_button.click()
            self.accepted_cookies = True

        leaderboard_rows = self.driver.find_elements_by_css_selector(
            "tr.competition-leaderboard__row"
        )

        if len(leaderboard_rows) < 2:
            return None  # no way to determine if we should minimize or maximize

        best_score = self._extract_public_leaderboard_score(leaderboard_rows[0])
        for i in range(len(leaderboard_rows) - 1):
            consecutive_places_diff = (
                self._extract_public_leaderboard_score(leaderboard_rows[i]) -
                self._extract_public_leaderboard_score(leaderboard_rows[i + 1])
            )
            if abs(consecutive_places_diff) > 1e-6:
                return "minimize" if consecutive_places_diff < 0 else "maximize"

        leaderboard_df = self._load_full_leaderboard()
        # For some annoying reason, leaderboard csv is sorted by team id and not by score
        if not (leaderboard_df["Score"] > best_score).any():
            return "maximize"
        elif not (leaderboard_df["Score"] < best_score).any():
            return "minimize"
        return None


test_links = [
    ("https://www.kaggle.com/c/titanic/", "maximize"),
    ("https://www.kaggle.com/c/jane-street-market-prediction/", "maximize"),
    ("https://www.kaggle.com/c/competitive-data-science-predict-future-sales/", "minimize"),
    ("https://www.kaggle.com/c/gan-getting-started/", "minimize"),
]

if __name__ == "__main__":
    with CompetitionScraper() as scraper:
        for link, true_result in test_links:
            my_result = scraper.get_optimization_type(link)
            assert my_result == true_result, (my_result, link)

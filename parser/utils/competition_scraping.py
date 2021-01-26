import tempfile
import zipfile

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from common import download_file


class CompetitionScraper:
    def __init__(self, max_load_wait=15.0):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--ignore-certificate-errors")
        self.options.add_argument("--incognito")
        self.options.add_argument("--headless")
        self.options.add_argument("window-size=1400,600")

        # Yes, I created a new account for this...
        self.acc_email = "leburner010203@gmail.com"
        self.acc_pwd = "12345678"

        self.max_load_wait = max_load_wait


    def __enter__(self):
        self.driver = webdriver.Chrome(options=self.options)
        _ = self.driver.implicitly_wait(self.max_load_wait)

        self.accepted_cookies = False
        self.logged_in = False
        return self


    def __exit__(self, exc_type, ecx_value, exc_traceback):
        self.driver.close()


    def _extract_public_leaderboard_score(self, leaderboard_row):
        score_block = leaderboard_row.find_element_by_xpath(
            "./td[contains(@data-th, 'Score')]"
        )
        return float(score_block.text)


    def _log_in(self):
        prev_page = self.driver.current_url

        self.driver.get("https://www.kaggle.com/account/login?phase=emailSignIn")

        fields = self.driver.find_elements_by_css_selector("input")
        fields[0].send_keys(self.acc_email)
        fields[1].send_keys(self.acc_pwd)

        sign_in_button = self.driver.find_element_by_xpath(
            "//span[contains(@class, 'cyTXUp')]"
        )
        sign_in_button.click()

        self.driver.get(prev_page)


    def _load_full_leaderboard(self):
        if not self.logged_in:
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

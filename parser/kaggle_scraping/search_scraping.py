from csv import writer
from io import StringIO
from time import sleep

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from base_scraper import BaseScraper


KAGGLE_LINK = "https://www.kaggle.com/"


class SearchScraper(BaseScraper):
    def __init__(self, max_load_wait=15.0):
        super().__init__(max_load_wait)


    def _get_all_notebooks_current_page(self, csv_writer):
        notebook_entries = self.driver.find_elements_by_xpath(
            "//a[contains(@class, 'sc-pIJmg')]"
        )

        for entry in notebook_entries:
            collected_data = []
            collected_data.append(entry.get_attribute("href")[len(KAGGLE_LINK):])

            data_containers = entry.find_elements_by_xpath(
                "./li/div[2]/div"
            )
            collected_data.append(data_containers[1].text)
            collected_data.append(data_containers[2].text[3:])

            data_containers = data_containers[3].find_elements_by_xpath(
                "./div/div"
            )

            collected_data.append(int(data_containers[3].text.split()[-1]))

            csv_writer.writerow(collected_data)

        return len(notebook_entries)


    def get_all_notebooks(self, lang, approx_notebook_count, chunk_size=1000):
        data = StringIO()
        csv_writer = writer(data)

        columns = ["ref", "title", "author", "totalVotes"]
        csv_writer.writerow(columns)

        lang = lang.capitalize()
        search_result_link = f"{KAGGLE_LINK}search?q=in%3Anotebooks+kernelLanguage%3A{lang}+sortBy%3Adate"

        self.driver.get(search_result_link)

        notebook_count = 0
        is_done = False
        while not is_done:
            notebooks_added = self._get_all_notebooks_current_page(csv_writer)
            notebook_count += notebooks_added
            if notebook_count >= approx_notebook_count:
                is_done = True

            buttons = self.driver.find_elements_by_xpath(
                "//button[contains(@class, 'mulrx')]"
            )
            if len(buttons) == 0:
                is_done = True
            buttons[0].click()
            sleep(5)

        data.seek(0)
        return pd.read_csv(data)


if __name__ == "__main__":
    with SearchScraper() as scraper:
        print(scraper.get_all_notebooks("python", 30).head())

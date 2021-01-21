from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class SourceScraper:
    def __init__(self, max_load_wait=15.0):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--ignore-certificate-errors")
        self.options.add_argument("--incognito")
        self.options.add_argument("--headless")
        self.options.add_argument("window-size=1400,600")

        self.max_load_wait = max_load_wait


    def __enter__(self):
        self.driver = webdriver.Chrome(options=self.options)
        _ = self.driver.implicitly_wait(self.max_load_wait)

        self.accepted_cookies = False
        return self


    def __exit__(self, exc_type, ecx_value, exc_traceback):
        self.driver.close()


    def get_source_links(self, kernel_link):
        self.driver.get(kernel_link)

        # Close cookie warning, if it is there
        if not self.accepted_cookies:
            accept_cookies_button = self.driver.find_element_by_xpath(
                "//*[@id='site-container']/div/div[6]/div/div[2]/div"
            )
            accept_cookies_button.click()
            self.accepted_cookies = True

        # Now onto the data source scraping
        sources_list = self.driver.find_element_by_xpath(
            "//ul[contains(@class, 'iLvHOz')]"
        )

        # Close all data source content lists
        open_spoiler_buttons = sources_list.find_elements_by_xpath(
            ".//i[contains(text(), 'arrow_drop_down')]"
        )
        for button in open_spoiler_buttons:
            button.click()

        # When content lists are closed, list of sources is correct and stable
        sources = sources_list.find_elements_by_xpath(".//p")

        links = []
        for i, source in enumerate(sources):
            source.click()

            source_thumbnail = self.driver.find_element_by_xpath("//div[contains(@class, 'sc-khlDuY')]")
            link_container = source_thumbnail.find_element_by_css_selector("a")

            links.append(link_container.get_attribute("href"))

        return links
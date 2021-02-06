# from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_source_links(webdriver, kernel_link):
    webdriver.driver.get(kernel_link)
    # Close cookie warning, if it is there
    webdriver.accept_cookies()
    # Now onto the data source scraping
    sources_list = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul>ul"))
    )

    # Close all data source content lists
    open_spoiler_buttons = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
        EC.presence_of_all_elements_located((By.XPATH, ".//i[contains(text(), 'arrow_drop_down')]"))
    )
    for button in open_spoiler_buttons:
        button.click()

    # When content lists are closed, list of sources is correct and stable
    sources = sources_list.find_elements_by_css_selector("p")

    links = []
    for i, source in enumerate(sources):
        source.click()

        source_thumbnail = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.sc-kcQtlj"))
        )
        link_container = source_thumbnail.find_element_by_css_selector("a")

        links.append(link_container.get_attribute("href"))

    return links

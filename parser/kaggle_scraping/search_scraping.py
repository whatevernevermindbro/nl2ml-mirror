from csv import writer
from io import StringIO

# import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from kaggle_scraping.conditions import element_disappeared, found_n_elements


KAGGLE_LINK = "https://www.kaggle.com/"


def _get_all_notebooks_current_page(webdriver, csv_writer, max_notebooks):
    try:
        notebook_entries = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
            found_n_elements((By.CSS_SELECTOR, "a.sc-qOvHb"), 20)
        )
    except TimeoutException:
        return 0

    notebook_count = 0
    for entry in notebook_entries:
        collected_data = []
        collected_data.append(entry.get_attribute("href")[len(KAGGLE_LINK):])

        csv_writer.writerow(collected_data)

        notebook_count += 1
        if notebook_count >= max_notebooks:
            break

    return notebook_count


def wait_loading_screen(webdriver):
    _ = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
        element_disappeared((By.CSS_SELECTOR, "div.sc-pQrUA"))
    )


def move_to_page(webdriver, page_id):
    # move to page 6, so we can avoid the loop
    buttons = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.ePLZfR"))
    )
    buttons[-1].click()
    wait_loading_screen(webdriver)

    if page_id == 0:
        return

    buttons = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.ePLZfR"))
    )
    buttons[page_id + 1].click()
    wait_loading_screen(webdriver)


def write_kernels_to_file(fd, webdriver, lang, max_notebook_count, start, pbar):
    assert 0 <= start < 3

    csv_writer = writer(fd)

    columns = ["ref"]
    csv_writer.writerow(columns)

    lang = lang.capitalize()
    search_result_link = f'{KAGGLE_LINK}search?q=in%3Anotebooks+kernelLanguage%3A{lang}+tag%3Anlp'

    webdriver.driver.get(search_result_link)

    move_to_page(webdriver, start)
    notebook_count = 0
    is_done = False
    while not is_done:
        notebooks_added = _get_all_notebooks_current_page(
            webdriver,
            csv_writer,
            max_notebook_count - notebook_count
        )

        notebook_count += notebooks_added
        pbar.update(notebooks_added)
        if notebooks_added == 0 or notebook_count >= max_notebook_count:
            is_done = True
            continue

        buttons = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.ePLZfR"))
        )

        button_id = 4
        if len(buttons) <= button_id:
            is_done = True
            continue
        buttons[button_id].click()

        wait_loading_screen(webdriver)

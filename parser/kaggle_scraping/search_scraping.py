from csv import writer
from io import StringIO

from time import sleep

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from kaggle_scraping.conditions import element_disappeared, found_n_elements


KAGGLE_LINK = "https://www.kaggle.com/"


def _get_all_objects_current_page(webdriver, csv_writer, max_objects):
    try:
        object_entries = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
            found_n_elements((By.CSS_SELECTOR, "a.sc-qOvHb"), 20)
        )
    except TimeoutException:
        return 0

    object_count = 0
    for entry in object_entries:
        collected_data = []
        collected_data.append(entry.get_attribute("href")[len(KAGGLE_LINK):])

        csv_writer.writerow(collected_data)

        object_count += 1
        if object_count >= max_objects:
            break

    return object_count


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


def build_search_link(object_type, filters):
    link = f"{KAGGLE_LINK}search?q=in%3A{object_type}"
    if filters is not None:
        for filter_name, value in filters.items():
            link += f"+{filter_name}%2A{value}"
    return link


def search_and_write_to_file(fd, webdriver, object_type, max_object_count, pbar, process_id, total_processes, filters=None):
    assert 0 <= process_id < total_processes

    csv_writer = writer(fd)

    columns = ["ref"]
    csv_writer.writerow(columns)

    search_result_link = build_search_link(object_type, filters)

    webdriver.driver.get(search_result_link)
    cur_page = 1
    if total_processes > 1:
        move_to_page(webdriver, process_id)
        cur_page = 6 + process_id

    object_count = 0
    is_done = False
    while not is_done:
        objects_added = _get_all_objects_current_page(
            webdriver,
            csv_writer,
            max_object_count - object_count
        )

        object_count += objects_added
        pbar.update(objects_added)
        if objects_added == 0 or object_count >= max_object_count:
            is_done = True
            continue

        buttons = WebDriverWait(webdriver.driver, webdriver.max_load_wait).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.ePLZfR"))
        )

        if total_processes > 1:
            button_id = total_processes + 1
            cur_page += process_id
        else:
            button_id = 2
            if cur_page < 6:
                button_id = cur_page - 1
            cur_page += 1

        if len(buttons) <= button_id:
            is_done = True
            continue
        sleep(5)
        buttons[button_id].click()

        wait_loading_screen(webdriver)

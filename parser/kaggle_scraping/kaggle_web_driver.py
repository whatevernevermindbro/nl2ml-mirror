from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class KaggleWebDriver:
    def __init__(self, max_load_wait=15.0):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--ignore-certificate-errors")
        self.options.add_argument("--incognito")
        self.options.add_argument("--remote-debugging-port=9222")
        self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        # self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("window-size=1400,600")

        # Yes, I created a new account for this...
        self.acc_email = "leburner010203@gmail.com"
        self.acc_pwd = "12345678"

        self.max_load_wait = max_load_wait

        self.driver = None
        self.accepted_cookies = False
        self.logged_in = False

    def __del__(self):
        if self.driver is not None:
            self.driver.quit()

    def load(self):
        self.driver = webdriver.Chrome(options=self.options)

        self.accepted_cookies = False
        self.logged_in = False

    def __enter__(self):
        self.load()
        return self

    def close(self):
        self.driver.quit()
        self.driver = None

    def __exit__(self, exc_type, ecx_value, exc_traceback):
        self.close()

    def log_in(self):
        if self.logged_in:
            return

        prev_page = self.driver.current_url

        self.driver.get("https://www.kaggle.com/account/login?phase=emailSignIn")

        fields = WebDriverWait(self.driver, self.max_load_wait).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input"))
        )
        fields[0].send_keys(self.acc_email)
        fields[1].send_keys(self.acc_pwd)

        sign_in_button = WebDriverWait(self.driver, self.max_load_wait).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "span.cyTXUp"))
        )
        sign_in_button.click()

        self.driver.get(prev_page)
        self.logged_in = True

    def accept_cookies(self):
        if self.accepted_cookies:
            return

        try:
            cookies_prompt = WebDriverWait(self.driver, self.max_load_wait).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.sc-qQwPu"))
            )
            accept_cookies_button = WebDriverWait(cookies_prompt, self.max_load_wait).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.sc-AxgMl.dhjQgX"))
            )
            accept_cookies_button.click()
        except TimeoutException:
            pass
        self.accepted_cookies = True

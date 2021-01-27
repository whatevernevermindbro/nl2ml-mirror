from selenium import webdriver


class BaseScraper:
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


    def _log_in(self):
        if self.logged_in:
            return

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
        self.logged_in = True


    def _accept_cookies(self):
        if self.accepted_cookies:
            return

        accept_cookies_button = self.driver.find_element_by_xpath(
            "//*[@id='site-container']/div/div[6]/div/div[2]/div"
        )
        accept_cookies_button.click()
        self.accepted_cookies = True

class element_disappeared(object):
    def __init__(self, locator):
        self.locator = locator

    def __call__(self, driver):
        elements = driver.find_elements(*self.locator)
        return len(elements) == 0


class found_n_elements(object):
    def __init__(self, locator, n):
        self.locator = locator
        self.n = n

    def __call__(self, driver):
        elements = driver.find_elements(*self.locator)
        if len(elements) >= self.n:
            return elements
        return False

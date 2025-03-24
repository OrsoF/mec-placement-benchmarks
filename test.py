from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import random

rnd_letters = "abcdefghijklmnopqrstuvwxyz1234567890"

# Installer et configurer ChromeDriver automatiquement
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Mode sans interface graphique

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)

# url = "https://streamable.com/gwvsjs"

for _ in range(10000):
    url = "https://streamable.com/" + "".join(
        random.choices(rnd_letters, k=5)
    ) 

    # url = "https://streamable.com/gwvsjs"

    driver.get(url)
    page_content = driver.page_source

    if len(page_content) > 10000:
        print(url)

driver.quit()

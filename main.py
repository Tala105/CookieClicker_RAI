import os
import json
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.service import Service
from CNN.agente import Agent
from Constants import CHECKPOINT_FILE, BUILDING_COSTS, UPGRADE_COSTS, BUILDING_IDS, UPGRADES_IDS, UPGRADE_COSTS_GROWTH


def click_on_element(driver, by_type, element_id, wait_time = 15, parent_div: WebElement = None):
    if not parent_div:
        WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((by_type, element_id))
        ).click()
    else:
        WebDriverWait(parent_div, wait_time).until(
            lambda d: parent_div.find_element((by_type, element_id))
        ).click()

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    buildings_cost = BUILDING_COSTS.copy()
    upgrades_cost = UPGRADE_COSTS.copy()
    upgrades_costs_growth = UPGRADE_COSTS_GROWTH.copy()
    upgrade_ids = UPGRADES_IDS.copy()
    cookies = 0
    """State format:
    [total_cookies, cookies_per_second, building_1_cps, building_1_cost, upgrade_1_cost,
    building_2_cps, building_2_cost, upgrade_2_cost, ...]
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as file:
            content = file.read().strip().splitlines()
            checkpoint_name = None

            for line in content:
                if line.startswith('model_checkpoint_path:'):
                    checkpoint_name = line.split(':')[1].strip().strip('"')
                    history_file_name = "CNN/Metadata_saved_files/" + checkpoint_name[:-3] + '_metadata.json'
                    print(f"Found history file: {history_file_name}")
                    print(f"Found checkpoint: {checkpoint_name}")

    if history_file_name and os.path.exists(history_file_name):
        checkpoint_content = json.load(open(history_file_name, 'r'))
        sequence = checkpoint_content['best_sequence']
        print(f"Best sequence: {sequence}")
    buying_sequence = []
    for element in sequence:
        buying_sequence.extend([int(element[0])]*int(element[2]))
    print(f"Buying sequence: {buying_sequence}")

    service = Service(executable_path="C:/Users/marcu/.cache/selenium/chromedriver/win64/136.0.7103.49/chromedriver.exe")
    options = Options()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://orteil.dashnet.org/cookieclicker/")
    driver.implicitly_wait(5)
    click_on_element(driver, By.ID, "langSelect-EN")
    sleep(2)
    upgrades = driver.find_element(By.ID, "upgrades")
    for element in buying_sequence:
        current_target = int(element)
        print(f"Current target: {current_target}")

        if current_target < len(buildings_cost):
            current_cost = buildings_cost[current_target-1]
            target_id = BUILDING_IDS[current_target-1]
        else:
            target_id = upgrade_ids[current_target - len(buildings_cost)].pop(0)
            current_cost = upgrades_cost[current_target - len(buildings_cost)-1]
            
            
        while current_cost > cookies:
            click_on_element(driver, By.ID, "bigCookie")
            cookies = driver.find_element(By.ID, "cookies").text.split(" ")[0].replace(",", "")
            cookies = int(cookies)
        if current_target < len(buildings_cost):
            click_on_element(driver, By.ID, target_id)
            for i in range(10):
                click_on_element(driver, By.ID, "bigCookie")
            buildings_cost[current_target-1] = int(driver.find_element(By.ID, target_id[:-1] + "Price" + target_id[-1]).text.split(" ")[0].replace(",", ""))
        else:
            element = WebDriverWait(driver, 10).until(
                lambda d: upgrades.find_element(By.CSS_SELECTOR, f'[data-id="{target_id}"]')
                ).click()
            upgrades_cost[current_target - len(buildings_cost)-1] *= upgrades_costs_growth[current_target - len(buildings_cost)].pop(0)
        cookies = 0
        
        

        

if __name__ == "__main__":
    main()

    
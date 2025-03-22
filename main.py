from CNN.agente import Agent
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service


def main():
    agent = Agent()
    checkpoint_file = 'CNN/Metadata_saved_files/checkpoint'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            content = file.read().strip().splitlines()
            checkpoint_name = None

            for line in content:
                if line.startswith('model_checkpoint_path:'):
                    checkpoint_name = line.split(':')[1].strip().strip('"')
                    print(f"Found checkpoint: {checkpoint_name}")

            latest_checkpoint = os.path.join("CNN/Metadata_saved_files", checkpoint_name)

    if latest_checkpoint and os.path.exists(latest_checkpoint + '.index'):
        agent.load(latest_checkpoint)

    service = Service(GeckoDriverManager().install())
    options = Options()
    options.headless = False
    driver = webdriver.Firefox(service=service, options=options)
    driver.get("https://orteil.dashnet.org/cookieclicker/")
    driver.implicitly_wait(5)


    
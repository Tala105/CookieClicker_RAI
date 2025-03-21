import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from CNN.agente import Agent

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep

import numpy as np
from math import ceil

tf.compat.v1.disable_eager_execution()

cursor_ids = range(7)
grandma_ids = range(7, 10)
farm_ids = range(10, 13)
mine_ids = range(16, 19)
factory_ids = range(13, 16)
ids = [cursor_ids, grandma_ids, farm_ids, mine_ids, factory_ids]



class CookieClickerBot:
    def __init__(self, agent):
        self.agent = agent
        self.options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get('http://orteil.dashnet.org/cookieclicker/')
        sleep(2)  # Espera a página carregar
        # Selecionar idioma
        self.select_language()
        sleep(2)
        self.open_stats_panel()
        
        self.cps = [0.1, 1, 8, 47, 260]
        self.bcost = [15, 100, 1100, 12000, 130000]
        self.bamount = [0, 0, 0, 0, 0]
        self.ucost = [100, 500, 3000, 10000, 40000]
        self.uamount = [0, 0, 0, 0, 0]
        self.iucost = [100, 500, 3000, 10000, 40000]


    def select_language(self):
        # Clicar na opção de selecionar idioma
        lang_select_button = self.driver.find_element(By.ID, 'langSelect-EN')
        lang_select_button.click()
        sleep(3)  # Espera a página recarregar com o idioma selecionado

    def click_cookie(self):
        cookie = self.driver.find_element(By.ID, 'bigCookie')
        cookie.click()

    def buy_item(self, item_id):
        try:
            if 'product' in item_id:
                item = self.driver.find_element(By.ID, item_id)
                item.click()
                self.bamount[int(item_id[-1])] += 1
                self.bcost[int(item_id[-1])] = ceil(self.bcost[int(item_id[-1])]*1.15)
            else:
                item = self.driver.find_element(By.CSS_SELECTOR, f"[data-id='{item_id}']")
                item.click()
                self.cps[int(item_id[-1])] *= 2
                self.uamount[int(item_id[-1])] += 1
                self.ucost[int(item_id[-1])] *= 5 if self.uamount[int(item_id[-1])] == 0 else 10
        except:
            self.click_cookie()
    def open_stats_panel(self):
        stats_button = self.driver.find_element(By.ID, 'statsButton')
        stats_button.click()
        sleep(0.5)  # Espera o painel abrir

    def get_total_cookies_made(self):
        stats = self.driver.find_element(By.ID, 'menu')
        stats_text = stats.text.split('\n')
        total_cookies_made = int(stats_text[6].replace(',', ''))
        return total_cookies_made

    def get_game_state(self):
        state = []
        
        cookies_text = self.driver.find_element(By.ID, 'cookies').text.split('\n')
        cookies_stored = int(cookies_text[0].split(' ')[0].replace(',', ''))
        cps = float(cookies_text[1].split(' ')[-1].replace(',', ''))
        cookies_made = self.get_total_cookies_made()
        
        state.append(cookies_made)
        state.append(cps)
        state.append(cookies_stored)
        
        for i in range(5):
            cps_building = cookies_stored - self.cps[i]
            cost_building = cookies_stored - self.bcost[i]
            cost_upgrade = cookies_stored - self.ucost[i]
                
            state.append(cps_building)
            state.append(cost_building)
            state.append(cost_upgrade)

        state = np.reshape(state, [1, len(state)])
        return state

    def play(self):
        NUM_BUILDINGS = 5
        state = np.zeros((1, 18))
        while state[0][0] < 10000000:
            state = self.get_game_state()
            action = self.agent.act(state)
            print(action)

            if action < NUM_BUILDINGS:
                item_id = f'product{action}'
                item_cost = self.bcost[action]
            else:
                item_id = f'upgrade{ids[action - NUM_BUILDINGS][0]}'
                ids[action - NUM_BUILDINGS] = ids[action - NUM_BUILDINGS][1:]
                item_cost = self.ucost[action - NUM_BUILDINGS]
                item = self.driver.find_element(By.CSS_SELECTOR, f"[data-id='{item_id}']")

            while state[0][2] < item_cost:
                self.click_cookie()
                state = self.get_game_state()

            self.buy_item(item_id)
            sleep(0.05)
        stats = self.driver.find_element(By.ID, 'menu')
        stats_text = stats.text.split('\n')
        time = stats_text[10].split(' ')[2]
        print(f'Objetivo alcançado em {time} segundos')
        self.driver.quit()

if __name__ == '__main__':
    state_size = 18  # Ajuste conforme necessário
    buy_size = 10  # Ajuste conforme necessário
    agent = Agent(state_size,  buy_size, training=False)
    agent.load("cookie.h5")
    bot = CookieClickerBot(agent)
    bot.play()

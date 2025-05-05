import numpy as np
import pygame
from Constants import NAMES, NUM_BUILDINGS, BUILDING_COSTS, CPS, UPGRADE_COSTS, GOAL
from CookieClicker.building import Building
from CookieClicker.upgrade import Upgrade

class Game:
    def __init__(self, screensize, fps, agente, render=False):
        self.screenSize = screensize
        self.fps = fps
        self.time = 0
        self.cookies = 0
        self.total = 0
        self.cps = 0
        self.click_power = 1
        self.buildings = []
        self.upgrades = []
        self.total_history = []
        self.action_history = []
        self.max_total = 10 ** 6
        self.max_cps = 10 ** 3
        self.max_cookies = 3 * 10 ** 6
        self.max_building_cps = 3 * 10 ** 3
        self.max_building_cost = 10 ** 6
        self.max_upgrade_cost = 10 ** 6

        for i in range(NUM_BUILDINGS):
            building = Building((self.screenSize[0] - 500, 50 + i * 40), f"{NAMES[i]}", BUILDING_COSTS[i], CPS[i])
            upgrade = Upgrade((self.screenSize[0] - 250, 50 + i * 40), f"Up {NAMES[i]}", UPGRADE_COSTS[i], building)
            self.buildings.append(building)
            self.upgrades.append(upgrade)

        self.agent = agente
        self.render = render
        self.cookie_image = pygame.image.load("CookieClicker/images/cookie.png")
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screenSize)
            self.clock = pygame.time.Clock()
            self.draw()
            pygame.display.flip()

    def normalize_state(self):
        norm_state = [
            self.total / self.max_total,
            self.cps / self.max_cps,
        ]
        for i in range(NUM_BUILDINGS):
            norm_state.append(self.buildings[i].cps / self.max_building_cps)
            norm_state.append(self.buildings[i].cost / self.max_building_cost)
            norm_state.append(self.upgrades[i].cost / self.max_upgrade_cost)
        return norm_state

    def get_state(self):
        return self.normalize_state()

    def run(self):
        running = True
        while running:
            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or self.total >= GOAL:
                        running = False
                        return self.total_history
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.handleClick(event.pos, event.button == 3)

            state = self.get_state()
            state = np.reshape(state, [1, len(state)])
            action = self.agent.act(state)

            if action < NUM_BUILDINGS:
                bought = self.buildings[action]
            else:
                bought = self.upgrades[action - NUM_BUILDINGS]

            while self.cookies < bought.cost:
                self.step(0)
            print(bought.name)

            self.step(action)
            self.total_history.append(self.total)
            if self.render:
                pygame.display.flip()
                self.clock.tick(self.fps)

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.cookie_image, (0, 30))
        for building in self.buildings:
            font = pygame.font.Font(None, 36)
            text = font.render(str(building), True, (0, 0, 0))
            self.screen.blit(text, building.position)
        for upgrade in self.upgrades:
            font = pygame.font.Font(None, 36)
            text = font.render(str(upgrade), True, (0, 0, 0))
            self.screen.blit(text, upgrade.position)
        font = pygame.font.Font(None, 36)
        totaltext = f"Total: {round(self.total)}" if self.total < 10 ** 5 else f"Total: {self.total:.1E}"
        cookiestext = f"Cookies: {round(self.cookies)}" if self.cookies <= 10 ** 5 else f"Cookies: {self.cookies:.1E}"
        cpstext = f"CPS: {round(self.cps, 1)}" if self.cps < 100 \
            else f"CPS: {round(self.cps)}" if self.cps < 10 ** 5 else f"CPS: {self.cps:.1E}"
        text = font.render(f"{totaltext} \t {cookiestext} \t {cpstext}", True, (0, 0, 0))
        self.screen.blit(text, (0, 0))

    def handleClick(self, position, right_click):
        if position[0] < 150 and position[1] < 150:
            self.cookie_click()
        for building in self.buildings:
            if building.position[0] < position[0] < building.position[0] + 150 and building.position[1] < position[1] + 30:
                building.buy(self)
        for upgrade in self.upgrades:
            if upgrade.position[0] < position[0] < upgrade.position[0] + 150 and upgrade.position[1] < position[1] + 30:
                upgrade.buy(self)

    def cookie_click(self):
        self.cookies += self.click_power
        self.total += self.click_power

    def step(self, action):
        self.time += 1
        if action == 0:
            self.cookie_click()
        elif action <= NUM_BUILDINGS:
            self.buildings[action - 1].buy(self)
        else:
            self.upgrades[action - 1 - NUM_BUILDINGS].buy(self)
        self.cps = sum([building.amount * building.cps for building in self.buildings])
        self.cookies += self.cps / self.fps
        self.total += self.cps / self.fps
        for upgrade in self.upgrades:
            upgrade.update()
        if self.render:
            self.draw()
            pygame.display.flip()
        self.append_action(action)
        return self.get_state()

    def append_action(self, action):
        if action == 0:
            return
        if not self.action_history:
            self.action_history.append(f"{action}x1")
            return

        current_value, count = self.action_history[-1].split('x')
        if int(current_value) == action:
            new_count = int(count) + 1
            self.action_history[-1] = f"{current_value}x{new_count}"
            return

        self.action_history.append(f"{action}x1")

    def get_action_history(self):
        return self.action_history

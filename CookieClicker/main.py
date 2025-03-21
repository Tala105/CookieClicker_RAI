import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import matplotlib.pyplot as plt
import tensorflow as tf
from Constants import NUM_BUILDINGS
from CookieClicker.game import Game
from CNN.agente import Agent
from CNN.random_agent import RandomAgent
from CNN.spender_agent import SpenderAgent
tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    screenSize = (640, 480)
    fps = 60

    # Criação dos agentes
    agente1 = Agent(3 * NUM_BUILDINGS + 3, 2 * NUM_BUILDINGS, training=False)
    agente1.load("cookie.h5")
    agente2 = RandomAgent(2 * NUM_BUILDINGS)
    agente3 = SpenderAgent(3 * NUM_BUILDINGS + 3, 3, 2 * NUM_BUILDINGS)

    # Criação dos jogos
    game1 = Game(screenSize, fps, agente1)
    game2 = Game(screenSize, fps, agente2)
    game3 = Game(screenSize, fps, agente3)

    # Execução dos jogos
    agent_history = game1.run()
    print("*"*15 + "Agent done" + "*"*15)
    random_history = game2.run()
    print("*"*15 + "Random done" + "*"*15)
    spend_history = game3.run()
    print("*"*15 + "Spender done" + "*"*15)

    plt.plot(agent_history, label="Agent")
    plt.plot(random_history, label="Random")
    plt.plot(spend_history, label="Spender")
    plt.legend()
    plt.grid()
    plt.show()

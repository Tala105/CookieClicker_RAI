from math import ceil


class Building():
    def __init__(self, position, name, cost, cps):
        self.position = position
        self.name = name
        self.cost = cost
        self.cps = cps
        self.amount = 0
    
    def __str__(self):
        return f"{self.name} ({self.cost})" if self.cost <= 10**4 else f"{self.name} ({self.cost:.1E})"
    def __repr__(self):
        return f"{self.name} ({self.cost})"
    
    def buy(self, game):
        if game.cookies >= self.cost:
            game.cookies -= self.cost
            self.cost = ceil(self.cost*1.15)
            self.amount += 1
            return True
        return False

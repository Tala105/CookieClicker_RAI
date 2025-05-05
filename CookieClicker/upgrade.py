class Upgrade():
    def __init__(self, position, name, cost, building, multiplier=2):
        self.position = position
        self.name = name
        self.cost = cost
        self.building = building
        self.multiplier = multiplier
        self.amount = 0
        self.buyable = False

    def __str__(self):
        return f"{self.name} ({self.cost})" if self.cost <= 10**4 else f"{self.name} ({self.cost:.1E})"
    
    def __repr__(self):
        return f"{self.name} ({self.cost})"
    
    def buy(self, game):
        self.update()
        if self.buyable and game.cookies >= self.cost:
            game.cookies -= self.cost
            if self.building.name == "Cursor":
                game.click_power *= self.multiplier
            if self.amount == 0:
                self.cost *= 5
            else:
                self.cost *= 10
            self.amount += 1
            self.building.cps *= self.multiplier
            return True
        return False

    def update(self):
        cursor_amounts = {1:1, 2:1, 3:25, 4:50, 5:100}
        amounts = {1:1, 2:5, 3:25, 4:50, 5:100}
        if self.building.name == "Cursor":
            self.buyable = self.building.amount >= cursor_amounts[self.amount+1]
        else:
            self.buyable = self.building.amount >= amounts[self.amount+1] 

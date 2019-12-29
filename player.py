import random

class Player:

    def __init__(self, name):
        self.name = name
        self.wins = 0
        self.loss = 0

    def strategy(self, table, position):
        """Call table to move current position up or down or do nothing."""

    def strategy_diff(self, table, position, source, target):
        diff = source - target
        if diff < -table.unit:
            table.move_down(position)
        elif diff > table.unit:
            table.move_up(position)

class User(Player):

    def __init__(self, name='user'):
        super().__init__(name)

class AI(Player):

    def __init__(self, name='ai'):
        super().__init__(name)

    def strategy(self, table, position):
        self.strategy_diff(table, position, getattr(table, position).centery,
                           table.ball.y)

class Random(Player):

    def __init__(self, name='random'):
        super().__init__(name)

    def strategy(self, table, position):
        r = random.randrange(3)
        if r == 1:
            table.move_down(position)
        elif r == 2:
            table.move_up(position)

class Model(Player):

    def __init__(self, name='model'):
        super().__init__(name)
        self.target = None
        self.target_set = False

    def strategy(self, table, position):
        if position == 'left_pad':
            if table.ball.dx < 0:
                self.strategy_model(table, position)
            else:
                self.strategy_center(table, position)
        elif table.ball.dx > 0:
            self.strategy_model(table, position)
        else:
            self.strategy_center(table, position)

    def strategy_center(self, table, position):
        self.target_set = False
        self.strategy_diff(table, position, getattr(table, position).centery,
                           table.q_height)

    def strategy_model(self, table, position):
        if not self.target_set or (table.ball.x == table.q_width and
                                   table.ball.y == table.q_height):
            self.target = table.model(position)
            self.target_set = True
        self.strategy_diff(table, position, getattr(table, position).centery,
                           self.target)

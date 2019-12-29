class Game:

    def __init__(self, table, left_player, right_player, max_score=5):
        self.table = table
        self.left_score = 0
        self.right_score = 0
        self.rally = 0
        self.winner = None
        self.left_player = left_player
        self.right_player = right_player
        self.max_score = max_score
        self.service = 1

    def is_terminal(self):
        return self.winner is not None

    def check_winner(self):
        if self.left_score < self.max_score and self.right_score < self.max_score:
            return
        diff = abs(self.left_score - self.right_score)
        if diff > 1:
            self.winner = 0
        elif diff < 1:
            self.winner = 1

    def end(self):
        left = 1
        right = -1
        if self.winner:
            left, right = right, left
        self.left_player.wins += left
        self.right_player.loss += right

    def reset(self):
        self.rally = 0
        self.service *= -1
        self.table.reset(self.service)
        for _ in range(5):
            self.advance_players()

    def increment_rally(self):
        self.rally += 1
        if self.rally % 2:
            self.table.speed_up()
            # print('speed up:', self.table.ball.dt)

    def advance_players(self):
        self.left_player.strategy(self.table, 'left_pad')
        self.right_player.strategy(self.table, 'right_pad')

    def advance_ball(self):
        for flag in self.table.advance_ball():
            if flag < 2:
                self.increment_rally()
                continue
            if flag == 2:
                self.right_score += 1
            else:
                self.left_score += 1
            self.check_winner()
            return True
        return False

    def advance(self):
        self.advance_players()
        return self.advance_ball()

    def play(self):
        while True:
            if self.advance():
                if self.is_terminal():
                    break
                self.reset()
        self.end()

from player import Player, AI, Random
from game import Game
from table import Table
from ui import UI
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import time
import os

matplotlib.rc('axes.formatter', useoffset=False)

DATA_PATH = os.getcwd() + '/data/'

class QPlayer(Player):

    # x[0,31]+2  y[0,15]  ly[0,15] ry[0,15] dx{-1,1}  dy{-1,1}  dt[1]
    states = 524290 # 2 + 32*16*16*16*2*2
    actions = 3 # 0 null, 1 up, 2 down
    rewards = [-1, 1]

    def __init__(self, **kwargs):
        name = 'q-' + '-'.join(str(x) for kv in kwargs.items() for x in kv)
        super().__init__(name)
        try:
            self.qtable = np.load(DATA_PATH + self.name + '.npy')
        except FileNotFoundError:
            self.qtable = None
            self.set_qtable()
        assert 'alpha' in kwargs
        assert 'gamma' in kwargs
        assert 'position' in kwargs
        self.__dict__.update(kwargs)
        self.reward = 1 if self.position == 'left' else -1
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

    def set_qtable(self):
        self.qtable = np.zeros((self.states+1, self.actions))

    def save_qtable(self):
        np.save(DATA_PATH + self.name + '.npy', self.qtable)
        print('saved:\n%s' % (DATA_PATH + self.name + '.npy'))

    def episodes(self):
        return self.qtable[0][0]

    def increment_episodes(self):
        self.qtable[0][0] += 1

    # x[0,31]+2  y[0,15]  ly[0,15] ry[0,15] dx{-1,1}  dy{-1,1}  dt[1]
    @classmethod
    def flatten_grid_count(cls, x, y, left, right, dx, dy):
        x //= 32
        if x < 0:
            x = 0
        elif x > 31:
            x = 31
        y //= 16
        left //= 16
        right //= 16
        dx = 0 if dx <= 0 else 1
        dy = 0 if dy <= 0 else 1
        return int(2*(16*(16*(16*x + y) + left) + dx) + dy)

    @classmethod
    def state(cls, table):
        if table.ball.x <= table.ball.r:
            return 1
        if table.ball.x >= table.width + table.ball.r:
            return 2
        return 3 + cls.flatten_grid_count(table.ball.x, table.ball.y,
            table.left_pad.centery, table.right_pad.centery, table.ball.dx, table.ball.dy)

    def strategy(self, table, position):
        assert position == self.position
        s = self.state(table)
        Q = self.qtable
        a = np.argmax(Q[s])
        if a == 1:
            table.move_up(self.position)
        elif a == 2:
            table.move_down(self.position)

    def state_to_reward(self, s):
        """Return reward for given state."""
        if s > 2:
            return 0
        return -self.reward if s == 1 else self.reward

    def get_decay(self):
        """Return decay function for epsilon greedy schedule."""
        return 1 / (1 + self.episodes())

    def exploration(self, ar):
        d = self.get_decay()
        # Prob(greedy) = 1 - d
        #              = 1 - d' + d'/k; k = self.actions
        # d = d * k/(k-1)
        d *= self.actions/(self.actions-1)
        greedy = np.random.random() >= d
        return np.argmax(ar) if greedy else np.random.randint(self.actions)

    def strategy_train(self, table):
        s = self.state(table)
        r = self.state_to_reward(s)
        Q = self.qtable

        # check if terminal state
        if r:
            Q[s][0] = r

        # check if starting state
        if self.prev_state is not None:
            ps = self.prev_state
            pa = self.prev_action
            pr = self.prev_reward

            # update previous state action reward
            Q[ps][pa] += (self.alpha *
                (pr + self.gamma * max(Q[s]) - Q[ps][pa]))

        # update new state action reward
        self.prev_state = s
        self.prev_action = a = self.exploration(Q[s])
        self.prev_reward = r

        # take action
        if a == 1:
            table.move_up(self.position)
        elif a == 2:
            table.move_down(self.position)

class QEpsilonGreedyPlayer(QPlayer):

    def __init__(self, alpha, gamma, epsilon, position):
        super().__init__(alpha=alpha, gamma=gamma, epsilon=epsilon, decay=1, position=position)

    def get_decay(self):
        result = self.decay
        self.decay *= self.epsilon
        return result

class QLPlayer(Player):

    def __init__(self, qtable_left, qtable_right, name='ql'):
        super().__init__(name)
        self.qtable_left = qtable_left
        self.qtable_right = qtable_right

    def strategy(self, table, position):
        s = QPlayer.state(table)
        Q = getattr(self, '_'.join(('qtable', position)))
        a = np.argmax(Q[s])
        if a == 1:
            table.move_up(position)
        elif a == 2:
            table.move_down(position)

# left_name = 'q-alpha-1-gamma-1-epsilon-0.999-decay-1-position-left.npy'
# right_name = 'q-alpha-1-gamma-1-epsilon-0.999-decay-1-position-right.npy'
# ql = QLPlayer(np.load(DATA_PATH + left_name), np.load(DATA_PATH + right_name))


class Train:

    def __init__(self, table, left_player, right_player, max_score, max_sets,
                 stat_sets, display_sets):
        self.table = table
        self.left_player = left_player
        self.right_player = right_player
        self.service = 1
        self.speed_up_flag = True
        self.max_score = max_score
        self.max_sets = max_sets
        self.stat_sets = stat_sets
        self.display_sets = display_sets
        self.qplayers = [player for player in (left_player, right_player)
                         if isinstance(player, QPlayer)]
        self.non_qplayer_positions = [(player, position) for player, position
            in ((left_player, 'left_pad'), (right_player, 'right_pad'))
            if not isinstance(player, QPlayer)]

        self.name = '-'.join((self.left_player.name, 'vs',
                              self.right_player.name, 'training'))

        self.vars = ['score_diffs', 'left_hits', 'right_hits', 'speed_ups']

        try:
            self.data = np.load(DATA_PATH + self.name + '-data' + '.npy')
            self.set_num = self.data[-1][0]
            if self.max_sets >= self.data.shape[1]:
                d = self.max_sets - self.data.shape[1]
                print('adding d data positions')
                self.data = np.pad(self.data, ((0,0), (0,d)), mode='constant')
        except FileNotFoundError:
            self.data = np.zeros((len(self.vars)+1, self.max_sets), dtype=int)
            self.set_num = 0

        kwargs = {v: self.data[i] for i,v in enumerate(self.vars)}
        self.__dict__.update(kwargs)
        print('episodes %f' % self.qplayers[0].episodes())
        print('sets %d' % self.set_num)

    def save(self):
        self.save_qtables()
        self.save_data()

    def save_qtables(self):
        for player in self.qplayers:
            player.save_qtable()

    def save_data(self):
        self.data[-1][0] = self.set_num
        np.save(DATA_PATH + self.name + '-data' + '.npy', self.data)
        print('saved:\n%s' % (DATA_PATH + self.name + '-data' + '.npy'))

    def display(self):
        self.display_data()
        self.display_stats()

    def display_data(self):
        fig, ax = plt.subplots(nrows=len(self.vars), ncols=1, sharex=True)
        fig.suptitle(self.name + '-data')

        ax[3].set_xlabel('sets (sum of %d rallies)' % self.max_score)
        for i,v in enumerate(self.vars):
            ax[i].set_ylabel(v)

        x = range(1, self.set_num+1)
        for i,c in enumerate('kbrg'):
            ax[i].scatter(x, self.data[i][:self.set_num], marker='o', color=c)

        fig.set_size_inches(16, 12)
        plt.savefig(DATA_PATH + self.name + '-data' + '.png')
        print('saved:\n%s' % (DATA_PATH + self.name + '-data' + '.png'))
        plt.show()
        plt.close()

    def display_stats(self):
        fig, ax = plt.subplots(nrows=len(self.vars),
                               ncols=len(self.stat_sets),
                               sharex='all', sharey='row')
        fig.suptitle(self.name + '-stats')

        cap = 'Units: sum of %d rallies per set' % self.max_score
        fig.text(.5, .02, cap, fontsize=12, horizontalalignment='center')

        colors = list('kbrg')
        for i,v in enumerate(self.vars):
            ax[i][0].set_ylabel(v)
            data_arr = getattr(self, v)
            for j,s in enumerate(self.stat_sets):
                ax[3][j].set_xlabel('sets (prev %d)' % s)
                x = np.linspace(self.stat_sets[j], self.set_num,
                                num=min(10, self.set_num//self.stat_sets[j]),
                                dtype=int)
                y = [np.mean(data_arr[a-s: a]) for a in x]
                e = [np.std(data_arr[a-s: a]) for a in x]
                ax[i][j].errorbar(x, y, e, capsize=2, linestyle='-',
                                  marker='o', color=colors[i])

        fig.set_size_inches(16, 12)
        plt.savefig(DATA_PATH + self.name + '-stat' + '.png')
        print('saved:\n%s' % (DATA_PATH + self.name + '-stat' + '.png'))
        plt.show()
        plt.close()

    def reset(self):
        self.service *= -1
        self.table.reset(self.service)
        for _ in range(5):
            self.advance_players()

    def check_speed_up(self):
        if self.speed_up_flag:
            self.table.speed_up()
            self.speed_ups[self.set_num] += 1
        self.speed_up_flag = not self.speed_up_flag

    def advance_players(self):
        for player in self.qplayers:
            player.strategy_train(self.table)
        for player, position in self.non_qplayer_positions:
            player.strategy(self.table, position)

    def increment_episodes(self):
        for player in self.qplayers:
            player.increment_episodes()

    def advance_ball(self):
        for flag in self.table.advance_ball():
            if not flag:
                self.left_hits[self.set_num] += 1
                self.check_speed_up()
                continue
            if flag == 1:
                self.right_hits[self.set_num] += 1
                self.check_speed_up()
                continue
            if flag == 2:
                self.score_diffs[self.set_num] -= 1
                self.increment_episodes()
            else:
                self.score_diffs[self.set_num] += 1
                self.increment_episodes()
            self.advance_players()
            return True
        return False

    def advance(self):
        self.advance_players()
        return self.advance_ball()

    def train(self):
        while self.set_num < self.max_sets:
            for _ in range(self.max_score):
                while True:
                    if self.advance():
                        break
                self.reset()
            self.set_num += 1
            if self.set_num in self.display_sets:
                self.save()
                self.display()
                break
        #         if self.early_stop():
        #             break
        # self.save()


    def early_stop(self):
        if self.set_num == self.max_sets:
            return False
        cont = input('continue training this model? (y/n) ')
        cont = True if cont == '' or cont == 'y' or cont == 'Y' else False
        if cont:
            return False
        msg = 'early stop: %d sets for \n%s' % (self.set_num, self.name)
        print(msg)
        return True

    def play_game(self):
        table = Table()
        game = Game(table, self.left_player, self.right_player, max_score=10)
        ui = UI(game)
        ui.start()

def main():
    age = [(a,g,e) for a in (1.0, .95, .9, .7, .5)
                   for g in (1.0, .95, .9, .7, .5)
                   for e in (.999, .9, .95, .9, .7, .5)]
    random.shuffle(age)
    for a,g,e in age:
        t = time.time()
        table = Table()
        ql = QEpsilonGreedyPlayer(alpha=a, gamma=g, epsilon=e,
                                  position='left_pad')
        # r = Random()
        qr = QEpsilonGreedyPlayer(alpha=a, gamma=g, epsilon=e,
                                  position='right_pad')
        max_score = 10
        max_sets = 10000
        stat_sets = [10, 30, 100, 300]
        display_sets = [2000]

        print('train: alpha=%f, gamma=%f, epsilon=%f' % (a, g, e))
        T = Train(table, ql, qr, max_score, max_sets, stat_sets, display_sets)
        T.train()
        d = time.time() - t
        m, s = divmod(d, 60)
        print('time elapsed: %s min %s sec' % (int(m), int(s)))
        # T.play_game()

if __name__ == '__main__':
    main()

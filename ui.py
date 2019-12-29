import pygame
from pygame import Color

class UI:

    def __init__(self, game):
        self.game = game
        self.table = game.table

        pygame.init()
        pygame.display.set_caption('Pong')
        self.surface = pygame.display.set_mode((self.table.width, self.table.height))
        self.font = pygame.font.SysFont('monospace', self.table.unit*16)

        assert isinstance(self.table.ball.x, int), (self.table.ball.x)
        assert isinstance(self.table.ball.y, int), (self.table.ball.y)
        assert isinstance(self.table.ball.r, int), (self.table.ball.r)
        self.table.ball.rect = pygame.draw.circle(self.surface, Color('red'), (self.table.ball.x, self.table.ball.y), self.table.ball.r)

        self.clock = pygame.time.Clock()

    def start(self):
        self.pause_loop(1)
        self.main_loop()

    def reset(self):
        self.pause_loop(.5)
        self.game.reset()
        self.update()
        self.pause_loop(2.5)

    def pause_loop(self, sec):
        for _ in range(int(sec*60)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            self.key_press(pygame.key.get_pressed())
            self.update()
            self.clock.tick(60)

    def get_surf_array(self):
        return pygame.surfarray.array2d(self.surface).astype(bool)

    def main_loop(self):
        while True:
            if self.game.is_terminal():
                self.pause_loop(5)
                pygame.quit()
                return
                ### if left_wins do ... ###

            self.advance_frame()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.key_press(pygame.key.get_pressed())
            self.update()
            self.clock.tick(60)

            # print(self.get_surf_array())

    def advance_frame(self):
        flag = self.game.advance()
        self.update()
        if flag:
            self.reset()

    def update(self):
        self.surface.fill(Color('black'))
        self.draw_objects()
        pygame.display.flip()

    def draw_objects(self):
        pygame.draw.rect(self.surface, Color('white'), self.table.net)
        pygame.draw.rect(self.surface, Color('blue'), self.table.left_pad)
        pygame.draw.rect(self.surface, Color('white'), self.table.right_pad)
        pygame.draw.circle(self.surface, Color('red'), (int(self.table.ball.x), int(self.table.ball.y)), self.table.ball.r)
        self.draw_scoreboard()
        # self.draw_lines()

    def draw_lines(self):
        pygame.draw.line(self.surface, (255,255,0), (0,0), (1023,0), 1)
        pygame.draw.line(self.surface, (255,255,0), (0,0), (0,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (1023,0), (1023,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (0,511), (1023,511), 1)

        pygame.draw.line(self.surface, (255,255,0), (32,0), (32,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (991,0), (991,511), 1)

        pygame.draw.line(self.surface, (255,255,0), (0,32), (1023,32), 1)
        pygame.draw.line(self.surface, (255,255,0), (0,479), (1023,479), 1)
        pygame.draw.line(self.surface, (255,255,0), (479, 0), (479,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (544,0), (544,511), 1)

    def draw_scoreboard(self):
        # left score
        font = self.font.render(str(self.game.left_score), True, Color('blue'))
        rect = font.get_rect()
        rect.right = int(0.939453125 * self.table.q_width) # 481
        rect.top = 2 * self.table.unit # 28
        self.surface.blit(font, rect)

        # right score
        font = self.font.render(str(self.game.right_score), True, Color('white'))
        rect = font.get_rect()
        rect.left = int(1.05859375 * self.table.q_width) # 542
        rect.top = 2 * self.table.unit # 28
        self.surface.blit(font, rect)

        # rally shots
        font = self.font.render(str(self.game.rally), True, Color('red'))
        rect = font.get_rect()
        rect.right = int(0.939453125 * self.table.q_width) # 481
        rect.bottom = self.table.height - 1.5*self.table.unit # 489
        self.surface.blit(font, rect)

    def key_press(self, pressed):
        if pressed[pygame.K_w]:
            self.table.move_up('left_pad')
        if pressed[pygame.K_s]:
            self.table.move_down('left_pad')
        if pressed[pygame.K_UP]:
            self.table.move_up('right_pad')
        if pressed[pygame.K_DOWN]:
            self.table.move_down('right_pad')

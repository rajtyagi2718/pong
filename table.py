from pygame import Rect
from math import sin, cos, pi, sqrt
import random
from ball import Ball

class Table:

    def __init__(self, unit=8):
        self.unit = unit # 4
        self.q_width = 64*unit # 512
        self.q_height = 32*unit # 256
        self.width = 128*unit # 1024
        self.height = 64*unit # 512

        self.net = Rect(0, 0, unit*2, self.height) # (4, 512)
        self.net.topleft = (self.q_width - unit, 0)

        self.left_pad = Rect(0, 0, unit*4, unit*12) # (10, 64)
        self.left_pad.midleft = (unit*8, self.q_height) # (32, 256)

        self.right_pad = Rect(0, 0, unit*4, unit*12) # (10, 64)
        self.right_pad.midright = (self.width-unit*8, self.q_height) # (992, 256)

        self.start_speed = unit
        self.ball = Ball(self.q_width, self.q_height, *self.random_start_velocity(), self.start_speed, unit*2) # (512, 256, 1, 0, 4, 8)

    def move_up(self, pad):
        pad = getattr(self, pad)
        pad.top = max(0, pad.top - self.unit)

    def move_down(self, pad):
        pad = getattr(self, pad)
        pad.bottom = min(self.height, pad.bottom + self.unit)

    def move_ball(self):
        self.ball.x += self.ball.dx
        self.ball.y += self.ball.dy
        self.ball.update_rect()

    def hit_left_pad(self):
        return (self.ball.dx < 0 and self.ball.rect.colliderect(self.left_pad)
                and self.hit_pad_flag(self.left_pad, True))

    def hit_right_pad(self):
        return (self.ball.dx > 0 and self.ball.rect.colliderect(self.right_pad)
                and self.hit_pad_flag(self.right_pad, False))

    def hit_pad_flag(self, pad, flag):
        if pad.top-1 <= self.ball.y < pad.bottom+1:
            self.bounce_pad(pad)
        elif pad.left-1 <= self.ball.x < pad.right+1:
            if self.ball.y < pad.top:
                self.bounce_corner(1,-1) if flag else self.bounce_corner(-1,-1)
            else:
                self.bounce_corner(1, 1) if flag else self.bounce_corner(-1, 1)
        else:
            return self.hit_corner(pad)
        return True

    def hit_corner(self, pad):
        if self.ball.collide_bottom_right(pad):
            self.bounce_corner(-1, -1)
        elif self.ball.collide_bottom_left(pad):
            self.bounce_corner(1, -1)
        elif self.ball.collide_top_left(pad):
            self.bounce_corner(1, 1)
        elif self.ball.collide_top_right(pad):
            self.bounce_corner(-1, 1)
        else:
            return False
        return True

    def bounce_corner(self, x_sgn, y_sgn):
        r = random.random()
        theta = pi/4 + r * (pi/24)
        self.ball.dx = x_sgn * sin(theta)
        self.ball.dy = y_sgn * cos(theta)

    def bounce_pad(self, pad):
        diff = self.ball.y - pad.centery
        max_diff = self.unit*10
        max_deg = 37.5
        theta = (max_deg/180) * pi * (diff / max_diff)
        # print('bounce:', diff, 180*theta/pi)
        sgn = 1 if self.ball.dx < 0 else -1
        self.ball.dx = sgn * cos(theta)
        self.ball.dy = sin(theta)

    def hit_side(self):
        if ((self.ball.y <= self.ball.r and self.ball.dy < 0) or
            (self.ball.y >= self.height-self.ball.r and self.ball.dy > 0)):
            self.ball.dy *= -1
            return True
        return False

    def hit_left_base(self):
        return self.ball.x <= -self.ball.r

    def hit_right_base(self):
        return self.ball.x >= self.width + self.ball.r

    def advance_ball(self):
        result = []
        for _ in range(self.ball.dt):
            self.move_ball()
            self.hit_side()
            if self.hit_left_pad():
                # print('left paddle:', (round(self.x), round(self.y)))
                result.append(0)
            elif self.hit_right_pad():
                # print('right paddle:', (round(self.x), round(self.y)))
                result.append(1)
            elif self.hit_left_base():
                # print('left base:', (round(self.x), round(self.y)))
                result.append(2)
                break
            elif self.hit_right_base():
                # print('right base:', (round(self.x), round(self.y)))
                result.append(3)
                break
        return result

    def get_round_ball_center(self):
        return (round(self.ball.x, 2), round(self.ball.y, 2))

    def speed_up(self):
        self.ball.dt += self.unit // 4

    def reset(self, service):
        self.ball.x = self.q_width
        self.ball.y = self.q_height
        self.ball.dx, self.ball.dy = self.random_start_velocity()
        self.ball.dx *= service
        self.ball.dt = self.start_speed

    def random_start_velocity(self):
        theta = pi/6 * (random.random() - .5)
        return (cos(theta), sin(theta))


    def copy(self):
        T = Table(self.unit)
        T.left_pad.midleft = self.left_pad.midleft
        T.right_pad.midright = self.right_pad.midright
        T.ball.x = self.ball.x
        T.ball.y = self.ball.y
        T.ball.dx = self.ball.dx
        T.ball.dy = self.ball.dy
        T.ball.dt = self.ball.dt
        return T

    def model(self, position):
        if position == 'left_pad':
            assert self.ball.dx < 0
            result = self.model_left_pad()
        else:
            assert self.ball.dx > 0
            result = self.model_right_pad()
        return result

    def model_left_pad(self):
        targetx = self.left_pad.right + self.ball.r
        M = self.copy()
        M.left_pad.topleft = (0, 0)
        while M.ball.x > targetx:
            M.move_ball()
            M.hit_side()
        return M.ball.y

    def model_right_pad(self):
        targetx = self.right_pad.left - self.ball.r
        M = self.copy()
        M.right_pad.topleft = (0, 0)
        while M.ball.x < targetx:
            M.move_ball()
            M.hit_side()
        return M.ball.y

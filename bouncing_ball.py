import pygame
import sys
import numpy as np
import random

# Initialize Pygame and mixer
pygame.init()
pygame.mixer.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BALL_RADIUS = 24
BALL_COLOR = (100, 200, 255)  # Light blue planet
BALL_GLOW_COLOR = (100, 200, 255, 80)
BACKGROUND_COLOR = (10, 10, 30)  # Space blue-black
FPS = 60
STAR_COUNT = 100
STAR_COLOR = (255, 255, 255)
STAR_SPEED_RANGE = (1, 4)
TRAIL_LENGTH = 10

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Space Ball!")
clock = pygame.time.Clock()

# Ball properties
ball_x = WINDOW_WIDTH // 2
ball_y = WINDOW_HEIGHT // 2
ball_speed_x = 6
ball_speed_y = 4
ball_color = BALL_COLOR
trail = []

# Star particle system
def create_star():
    return {
        'x': random.randint(0, WINDOW_WIDTH),
        'y': random.randint(0, WINDOW_HEIGHT),
        'speed': random.uniform(*STAR_SPEED_RANGE),
        'size': random.randint(1, 3)
    }
stars = [create_star() for _ in range(STAR_COUNT)]

def update_stars():
    for star in stars:
        star['y'] += star['speed']
        if star['y'] > WINDOW_HEIGHT:
            star['x'] = random.randint(0, WINDOW_WIDTH)
            star['y'] = 0
            star['speed'] = random.uniform(*STAR_SPEED_RANGE)
            star['size'] = random.randint(1, 3)

def draw_stars():
    for star in stars:
        pygame.draw.circle(screen, STAR_COLOR, (int(star['x']), int(star['y'])), star['size'])

# Music: Generate a simple looping spacey melody
def generate_tone(frequency, duration_ms, volume=0.2):
    sample_rate = 44100
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = np.int16(tone * volume * 32767)
    # Make stereo by stacking the mono array
    stereo_audio = np.column_stack((audio, audio))
    return pygame.sndarray.make_sound(stereo_audio)

def play_background_music():
    notes = [220, 277, 330, 440, 554, 660, 880]
    melody = [random.choice(notes) for _ in range(8)]
    sounds = [generate_tone(f, 200) for f in melody]
    def play_loop():
        for s in sounds:
            s.play()
            pygame.time.wait(200)
        play_loop()
    import threading
    threading.Thread(target=play_loop, daemon=True).start()

# Fun sound effect
def play_bounce_sound():
    freq = random.choice([330, 440, 550, 660, 880])
    s = generate_tone(freq, 80, 0.4)
    s.play()

def draw_ball_with_glow(x, y, color):
    # Draw glow
    for i in range(8, 0, -1):
        alpha = int(20 * i)
        glow_surf = pygame.Surface((BALL_RADIUS*4, BALL_RADIUS*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, alpha), (BALL_RADIUS*2, BALL_RADIUS*2), BALL_RADIUS + i*3)
        screen.blit(glow_surf, (x-BALL_RADIUS*2, y-BALL_RADIUS*2), special_flags=pygame.BLEND_RGBA_ADD)
    # Draw trail
    for idx, (tx, ty, tcolor) in enumerate(trail):
        fade = int(255 * (1 - idx / TRAIL_LENGTH))
        pygame.draw.circle(screen, (*tcolor, fade), (int(tx), int(ty)), BALL_RADIUS)
    # Draw main ball
    pygame.draw.circle(screen, color, (int(x), int(y)), BALL_RADIUS)

def main():
    global ball_x, ball_y, ball_speed_x, ball_speed_y, ball_color, trail
    play_background_music()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if (ball_x - mx) ** 2 + (ball_y - my) ** 2 < BALL_RADIUS ** 2:
                    # Change color and speed randomly
                    ball_color = tuple(random.randint(100, 255) for _ in range(3))
                    ball_speed_x = random.choice([-1, 1]) * random.randint(3, 8)
                    ball_speed_y = random.choice([-1, 1]) * random.randint(3, 8)
                    play_bounce_sound()

        # Update ball position
        ball_x += ball_speed_x
        ball_y += ball_speed_y

        # Bounce off walls
        bounced = False
        if ball_x <= BALL_RADIUS or ball_x >= WINDOW_WIDTH - BALL_RADIUS:
            ball_speed_x = -ball_speed_x
            bounced = True
        if ball_y <= BALL_RADIUS or ball_y >= WINDOW_HEIGHT - BALL_RADIUS:
            ball_speed_y = -ball_speed_y
            bounced = True
        if bounced:
            play_bounce_sound()

        # Keep ball within screen bounds
        ball_x = max(BALL_RADIUS, min(WINDOW_WIDTH - BALL_RADIUS, ball_x))
        ball_y = max(BALL_RADIUS, min(WINDOW_HEIGHT - BALL_RADIUS, ball_y))

        # Update trail
        trail = [(ball_x, ball_y, ball_color)] + trail[:TRAIL_LENGTH-1]

        # Update and draw everything
        screen.fill(BACKGROUND_COLOR)
        update_stars()
        draw_stars()
        draw_ball_with_glow(ball_x, ball_y, ball_color)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main() 
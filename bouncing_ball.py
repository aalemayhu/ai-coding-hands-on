import pygame
import sys
import numpy as np
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

# Initialize Pygame and mixer with higher quality
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Folk music constants
FOLK_BPM = 120
FOLK_SAMPLE_RATE = 44100
FOLK_CHANNELS = 2
FOLK_SCALE = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
FOLK_ROOT_NOTE = 440  # A4
FOLK_CHORD_PROGRESSION = [
    [0, 4, 7],    # I
    [5, 9, 12],   # IV
    [7, 11, 14],  # V
    [2, 5, 9]     # vi
]

def generate_folk_note(frequency: float, duration: float, volume: float = 0.3) -> np.ndarray:
    """Generate a folk-style note with harmonics and slight detuning"""
    t = np.linspace(0, duration, int(FOLK_SAMPLE_RATE * duration), False)
    
    # Main note
    note = np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics for richer sound
    note += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)  # First harmonic
    note += 0.25 * np.sin(2 * np.pi * frequency * 3 * t)  # Second harmonic
    
    # Slight detuning for folk feel
    detune = 0.02
    note += 0.3 * np.sin(2 * np.pi * (frequency + detune) * t)
    
    # Apply envelope
    envelope = np.exp(-t * 2)  # Quick decay
    note *= envelope
    
    # Normalize and apply volume
    note = note / np.max(np.abs(note)) * volume
    
    # Convert to stereo
    stereo_note = np.column_stack((note, note))
    return np.int16(stereo_note * 32767)

def generate_folk_chord(root_freq: float, duration: float, volume: float = 0.2) -> np.ndarray:
    """Generate a folk-style chord with multiple notes"""
    t = np.linspace(0, duration, int(FOLK_SAMPLE_RATE * duration), False)
    chord = np.zeros_like(t)
    
    # Generate each note in the chord
    for interval in [0, 4, 7]:  # Major triad
        freq = root_freq * (2 ** (interval / 12))
        note = np.sin(2 * np.pi * freq * t)
        note += 0.5 * np.sin(2 * np.pi * freq * 2 * t)  # First harmonic
        chord += note
    
    # Apply envelope
    envelope = np.exp(-t * 1.5)  # Slower decay for chords
    chord *= envelope
    
    # Normalize and apply volume
    chord = chord / np.max(np.abs(chord)) * volume
    
    # Convert to stereo
    stereo_chord = np.column_stack((chord, chord))
    return np.int16(stereo_chord * 32767)

def generate_folk_melody() -> pygame.mixer.Sound:
    """Generate a complete folk melody loop"""
    melody_duration = 8  # 8 seconds per loop
    beat_duration = 60 / FOLK_BPM
    samples = []
    
    # Generate melody notes
    for i in range(16):  # 16 beats per loop
        if i % 4 == 0:  # Start of each bar
            # Play chord
            chord_root = FOLK_ROOT_NOTE * (2 ** (FOLK_SCALE[i % len(FOLK_SCALE)] / 12))
            chord = generate_folk_chord(chord_root, beat_duration * 2)
            samples.append(chord)
            
            # Play melody note
            note_freq = FOLK_ROOT_NOTE * (2 ** (FOLK_SCALE[(i + 2) % len(FOLK_SCALE)] / 12))
            note = generate_folk_note(note_freq, beat_duration * 2)
            samples.append(note)
        else:
            # Play melody notes
            note_freq = FOLK_ROOT_NOTE * (2 ** (FOLK_SCALE[(i + 4) % len(FOLK_SCALE)] / 12))
            note = generate_folk_note(note_freq, beat_duration)
            samples.append(note)
    
    # Combine all samples
    full_melody = np.concatenate(samples)
    return pygame.sndarray.make_sound(full_melody)

# Get screen info for fullscreen
screen_info = pygame.display.Info()
SCREEN_WIDTH = screen_info.current_w
SCREEN_HEIGHT = screen_info.current_h

# Base window size (for windowed mode)
BASE_WIDTH = 1280
BASE_HEIGHT = 720

# Scale factor for fullscreen
SCALE_FACTOR = min(SCREEN_WIDTH / BASE_WIDTH, SCREEN_HEIGHT / BASE_HEIGHT)

# Constants (scaled for fullscreen)
WINDOW_WIDTH = int(BASE_WIDTH * SCALE_FACTOR)
WINDOW_HEIGHT = int(BASE_HEIGHT * SCALE_FACTOR)
BALL_RADIUS = int(16 * SCALE_FACTOR)
PADDLE_WIDTH = int(160 * SCALE_FACTOR)
PADDLE_HEIGHT = int(56 * SCALE_FACTOR)
BLOCK_WIDTH = int(100 * SCALE_FACTOR)
BLOCK_HEIGHT = int(40 * SCALE_FACTOR)
BLOCK_PADDING = int(10 * SCALE_FACTOR)
BLOCK_ROWS = 6
BLOCK_COLS = 12
POWERUP_SIZE = int(30 * SCALE_FACTOR)
POWERUP_SPEED = 3
BALL_COLOR = (100, 200, 255)
BALL_GLOW_COLOR = (100, 200, 255, 80)
BACKGROUND_COLOR = (10, 10, 30)
FPS = 60
STAR_COUNT = int(150 * SCALE_FACTOR)
STAR_COLOR = (255, 255, 255)
STAR_SPEED_RANGE = (2, 6)
TRAIL_LENGTH = 15

# Game states
class GameState(Enum):
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2
    LEVEL_COMPLETE = 3

# Power-up types
class PowerUpType(Enum):
    WIDE_PADDLE = 0
    MULTI_BALL = 1
    SLOW_BALL = 2
    EXTRA_LIFE = 3
    LASER = 4

# Block types
class BlockType(Enum):
    NORMAL = 0
    HARD = 1
    POWERUP = 2
    INDESTRUCTIBLE = 3

# Colors for different block types
BLOCK_COLORS = {
    BlockType.NORMAL: [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50)],
    BlockType.HARD: [(150, 75, 75), (75, 150, 75), (75, 75, 150)],
    BlockType.POWERUP: [(200, 100, 200)],
    BlockType.INDESTRUCTIBLE: [(100, 100, 100)]
}

# Power-up colors
POWERUP_COLORS = {
    PowerUpType.WIDE_PADDLE: (255, 200, 50),
    PowerUpType.MULTI_BALL: (255, 100, 255),
    PowerUpType.SLOW_BALL: (100, 255, 100),
    PowerUpType.EXTRA_LIFE: (255, 50, 50),
    PowerUpType.LASER: (50, 200, 255)
}

# Tweening system
@dataclass
class Tween:
    start: float
    end: float
    duration: float
    current_time: float = 0
    easing_func: callable = None
    on_complete: callable = None
    delay: float = 0
    current_delay: float = 0

    def update(self, dt: float) -> Optional[float]:
        if self.current_delay < self.delay:
            self.current_delay += dt
            return None
        
        if self.current_time >= self.duration:
            if self.on_complete:
                self.on_complete()
            return self.end

        self.current_time += dt
        t = self.current_time / self.duration
        if self.easing_func:
            t = self.easing_func(t)
        return self.start + (self.end - self.start) * t

class TweenManager:
    def __init__(self):
        self.tweens: List[Tween] = []

    def add(self, tween: Tween):
        self.tweens.append(tween)

    def update(self, dt: float):
        self.tweens = [t for t in self.tweens if t.update(dt) is None]

# Easing functions
def ease_out_quad(t): return 1 - (1 - t) * (1 - t)
def ease_out_bounce(t):
    n1, d1 = 7.5625, 2.75
    if t < 1/d1: return n1*t*t
    elif t < 2/d1: return n1*(t-1.5/d1)**2 + 0.75
    elif t < 2.5/d1: return n1*(t-2.25/d1)**2 + 0.9375
    else: return n1*(t-2.625/d1)**2 + 0.984375
def ease_out_elastic(t):
    c4 = (2 * math.pi) / 3
    if t == 0 or t == 1: return t
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1
def ease_out_back(t):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

# Screen shake effect
class ScreenShake:
    def __init__(self):
        self.shake_amount = 0
        self.shake_decay = 0.9
        self.offset_x = 0
        self.offset_y = 0
        self.trauma = 0
        self.max_trauma = 1.0

    def shake(self, amount):
        self.trauma = min(self.trauma + amount, self.max_trauma)

    def update(self):
        if self.trauma > 0:
            self.shake_amount = self.trauma * self.trauma
            self.offset_x = random.uniform(-self.shake_amount, self.shake_amount) * 10
            self.offset_y = random.uniform(-self.shake_amount, self.shake_amount) * 10
            self.trauma = max(0, self.trauma - 0.1)
        else:
            self.offset_x = 0
            self.offset_y = 0

# Particle system with more effects
class Particle:
    def __init__(self, x, y, vx, vy, color, size, lifetime, gravity=0, rotation=0, rotation_speed=0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.gravity = gravity
        self.rotation = rotation
        self.rotation_speed = rotation_speed
        self.scale = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.lifetime -= 1
        self.rotation += self.rotation_speed
        self.scale *= 0.95
        return self.lifetime > 0

    def draw(self, surface):
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        color = (*self.color[:3], alpha)
        size = int(self.size * self.scale)
        
        # Draw particle with rotation
        if size > 0:
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (size, size), size)
            if self.rotation != 0:
                particle_surf = pygame.transform.rotate(particle_surf, self.rotation)
            surface.blit(particle_surf, (self.x - size, self.y - size))

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, count, color, size_range=(2, 4), speed_range=(1, 3), 
             lifetime_range=(20, 40), gravity=0, rotation_range=(-5, 5)):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*speed_range)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(*size_range)
            lifetime = random.randint(*lifetime_range)
            rotation = random.uniform(*rotation_range)
            self.particles.append(Particle(x, y, vx, vy, color, size, lifetime, gravity, 0, rotation))

    def emit_confetti(self, x, y, count=20):
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2  # Initial upward velocity
            color = random.choice(colors)
            size = random.uniform(3, 6)
            lifetime = random.randint(60, 120)
            rotation = random.uniform(-10, 10)
            self.particles.append(Particle(x, y, vx, vy, color, size, lifetime, 0.1, 0, rotation))

    def emit_shatter(self, x, y, color, count=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(4, 8)
            lifetime = random.randint(40, 80)
            rotation = random.uniform(-20, 20)
            self.particles.append(Particle(x, y, vx, vy, color, size, lifetime, 0.2, 0, rotation))

    def update(self):
        self.particles = [p for p in self.particles if p.update()]

    def draw(self, surface):
        for p in self.particles:
            p.draw(surface)

# Paddle with personality
class Paddle:
    def __init__(self):
        self.x = WINDOW_WIDTH // 2 - PADDLE_WIDTH // 2
        self.y = WINDOW_HEIGHT - 132  # Moved up 32px from original position (was -100)
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.color = (200, 200, 200)
        self.eye_color = (50, 50, 50)
        self.scale = 1.0
        self.rotation = 0
        self.blink_timer = 0
        self.blink_duration = 20
        self.is_blinking = False
        self.smile_scale = 0
        self.target_x = self.x
        self.last_ball_distance = float('inf')
        self.smile_velocity = 0  # For smooth smile animation

    def update(self, ball_x, ball_y):
        # Update target position (follow mouse with easing)
        mouse_x = pygame.mouse.get_pos()[0]
        self.target_x = mouse_x - self.width // 2
        self.target_x = max(0, min(WINDOW_WIDTH - self.width, self.target_x))
        
        # Ease towards target
        self.x += (self.target_x - self.x) * 0.2

        # Update blink timer
        if self.blink_timer > 0:
            self.blink_timer -= 1
            if self.blink_timer == 0:
                self.is_blinking = False
        elif random.random() < 0.02:  # 2% chance to blink each frame
            self.is_blinking = True
            self.blink_timer = self.blink_duration

        # Calculate ball distance and direction
        ball_distance = math.sqrt((ball_x - (self.x + self.width/2))**2 + 
                                (ball_y - (self.y + self.height/2))**2)
        
        # Calculate target smile scale based on ball distance
        # Smile more when ball is closer, with a smooth transition
        max_smile_distance = WINDOW_WIDTH * 0.4  # Increased range for smile
        target_smile = max(0, min(1, 1 - (ball_distance / max_smile_distance)))
        
        # Add some bounce to the smile when ball gets closer
        if ball_distance < self.last_ball_distance:
            target_smile *= 1.2  # Overshoot when ball is approaching
        
        # Smooth smile animation with spring physics
        smile_diff = target_smile - self.smile_scale
        self.smile_velocity += smile_diff * 0.3  # Spring force
        self.smile_velocity *= 0.8  # Damping
        self.smile_scale += self.smile_velocity
        
        # Clamp smile scale
        self.smile_scale = max(0, min(1.2, self.smile_scale))
        
        # Store current distance for next frame
        self.last_ball_distance = ball_distance

    def draw(self, surface):
        # Draw paddle body
        paddle_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(paddle_surf, self.color, (0, 0, self.width, self.height), border_radius=15)
        
        # Draw eyes (moved up slightly for taller paddle)
        eye_y = self.height * 0.25  # Moved eyes up
        eye_size = self.height * 0.2  # Slightly smaller eyes for taller paddle
        eye_spacing = self.width * 0.3
        
        if not self.is_blinking:
            # Draw eyes looking at ball
            pygame.draw.circle(paddle_surf, self.eye_color, 
                             (int(self.width * 0.3), int(eye_y)), int(eye_size))
            pygame.draw.circle(paddle_surf, self.eye_color, 
                             (int(self.width * 0.7), int(eye_y)), int(eye_size))
        else:
            # Draw blinking eyes
            pygame.draw.line(paddle_surf, self.eye_color,
                           (int(self.width * 0.3 - eye_size), int(eye_y)),
                           (int(self.width * 0.3 + eye_size), int(eye_y)), 3)
            pygame.draw.line(paddle_surf, self.eye_color,
                           (int(self.width * 0.7 - eye_size), int(eye_y)),
                           (int(self.width * 0.7 + eye_size), int(eye_y)), 3)

        # Draw smile with more dynamic behavior
        smile_height = self.height * 0.3 * self.smile_scale  # Increased smile height
        smile_width = self.width * 0.6
        smile_x = (self.width - smile_width) / 2
        smile_y = self.height * 0.5  # Moved smile down slightly
        
        # Draw a more expressive smile
        if self.smile_scale > 0.8:
            # Big smile with cheeks
            pygame.draw.arc(paddle_surf, self.eye_color,
                          (smile_x, smile_y, smile_width, smile_height * 2),
                          math.pi, 2 * math.pi, 3)
            # Add cheek details
            cheek_y = smile_y + smile_height * 0.5
            pygame.draw.circle(paddle_surf, self.eye_color,
                             (int(self.width * 0.2), int(cheek_y)), int(eye_size * 0.5))
            pygame.draw.circle(paddle_surf, self.eye_color,
                             (int(self.width * 0.8), int(cheek_y)), int(eye_size * 0.5))
        else:
            # Normal smile
            pygame.draw.arc(paddle_surf, self.eye_color,
                          (smile_x, smile_y, smile_width, smile_height * 2),
                          math.pi, 2 * math.pi, 2)

        # Apply scale and rotation
        if self.scale != 1.0 or self.rotation != 0:
            new_size = (int(self.width * self.scale), int(self.height * self.scale))
            paddle_surf = pygame.transform.scale(paddle_surf, new_size)
            if self.rotation != 0:
                paddle_surf = pygame.transform.rotate(paddle_surf, self.rotation)

        surface.blit(paddle_surf, (self.x, self.y))

class Block:
    def __init__(self, x: int, y: int, block_type: BlockType, color_index: int = 0):
        self.x = x
        self.y = y
        self.width = BLOCK_WIDTH
        self.height = BLOCK_HEIGHT
        self.block_type = block_type
        self.color_index = color_index
        self.hits = 0
        self.max_hits = 2 if block_type == BlockType.HARD else 1
        self.powerup_type = None
        if block_type == BlockType.POWERUP:
            self.powerup_type = random.choice(list(PowerUpType))
        self.scale = 1.0
        self.rotation = 0

    def hit(self) -> Optional[PowerUpType]:
        if self.block_type == BlockType.INDESTRUCTIBLE:
            return None
        
        self.hits += 1
        # Return powerup if block is destroyed (hits >= max_hits)
        if self.hits >= self.max_hits:
            return self.powerup_type
        # Return None but don't remove block yet
        return None

    def is_destroyed(self) -> bool:
        """Check if block should be removed"""
        return (self.block_type != BlockType.INDESTRUCTIBLE and 
                self.hits >= self.max_hits)

    def draw(self, surface):
        if self.block_type == BlockType.INDESTRUCTIBLE:
            color = BLOCK_COLORS[BlockType.INDESTRUCTIBLE][0]
        else:
            colors = BLOCK_COLORS[self.block_type]
            color = colors[self.color_index % len(colors)]
            if self.block_type == BlockType.HARD:
                # Darken color based on hits
                color = tuple(max(0, c - 50 * self.hits) for c in color)

        # Draw block with effects
        block_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw main block
        pygame.draw.rect(block_surf, color, (0, 0, self.width, self.height), 
                        border_radius=8)
        
        # Draw highlight
        highlight = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(highlight, (255, 255, 255, 30), 
                        (0, 0, self.width, self.height//3), 
                        border_radius=8)
        block_surf.blit(highlight, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw power-up indicator
        if self.block_type == BlockType.POWERUP and self.powerup_type:
            powerup_color = POWERUP_COLORS[self.powerup_type]
            pygame.draw.circle(block_surf, powerup_color,
                             (self.width//2, self.height//2),
                             self.width//4)

        # Apply scale and rotation
        if self.scale != 1.0 or self.rotation != 0:
            new_size = (int(self.width * self.scale), int(self.height * self.scale))
            block_surf = pygame.transform.scale(block_surf, new_size)
            if self.rotation != 0:
                block_surf = pygame.transform.rotate(block_surf, self.rotation)

        surface.blit(block_surf, (self.x, self.y))

class PowerUp:
    def __init__(self, x: float, y: float, powerup_type: PowerUpType):
        self.x = x
        self.y = y
        self.width = POWERUP_SIZE
        self.height = POWERUP_SIZE
        self.powerup_type = powerup_type
        self.vy = POWERUP_SPEED
        self.rotation = 0
        self.rotation_speed = random.uniform(-2, 2)
        self.scale = 1.0
        self.color = POWERUP_COLORS[powerup_type]

    def update(self):
        self.y += self.vy
        self.rotation += self.rotation_speed
        return self.y < WINDOW_HEIGHT

    def draw(self, surface):
        powerup_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw power-up icon
        if self.powerup_type == PowerUpType.WIDE_PADDLE:
            pygame.draw.rect(powerup_surf, self.color, 
                           (self.width*0.2, self.height*0.3, 
                            self.width*0.6, self.height*0.4))
        elif self.powerup_type == PowerUpType.MULTI_BALL:
            for i in range(3):
                pygame.draw.circle(powerup_surf, self.color,
                                 (self.width*(0.3 + i*0.2), self.height*0.5),
                                 self.width*0.2)
        elif self.powerup_type == PowerUpType.SLOW_BALL:
            pygame.draw.circle(powerup_surf, self.color,
                             (self.width*0.5, self.height*0.5), self.width*0.3)
            pygame.draw.line(powerup_surf, self.color,
                           (self.width*0.3, self.height*0.5),
                           (self.width*0.7, self.height*0.5), 3)
        elif self.powerup_type == PowerUpType.EXTRA_LIFE:
            pygame.draw.polygon(powerup_surf, self.color,
                              [(self.width*0.5, self.height*0.2),
                               (self.width*0.2, self.height*0.8),
                               (self.width*0.8, self.height*0.8)])
        elif self.powerup_type == PowerUpType.LASER:
            pygame.draw.rect(powerup_surf, self.color,
                           (self.width*0.3, self.height*0.2,
                            self.width*0.4, self.height*0.6))
            pygame.draw.polygon(powerup_surf, self.color,
                              [(self.width*0.5, self.height*0.8),
                               (self.width*0.3, self.height*0.6),
                               (self.width*0.7, self.height*0.6)])

        # Apply rotation
        if self.rotation != 0:
            powerup_surf = pygame.transform.rotate(powerup_surf, self.rotation)
        
        surface.blit(powerup_surf, (self.x, self.y))

class Ball:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.radius = BALL_RADIUS
        self.speed_x = 0
        self.speed_y = 0
        self.color = BALL_COLOR
        self.scale = 1.0
        self.rotation = 0
        self.stretch = 1.0
        self.is_laser = False
        self.laser_cooldown = 0
        self.min_speed = 6
        self.max_speed = 15

    def set_color(self, color):
        # Ensure color is a valid RGB tuple with integer values
        try:
            if isinstance(color, tuple) and len(color) >= 3:
                r, g, b = [int(c) for c in color[:3]]  # Convert to integers
                # Ensure values are in valid range (0-255)
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                self.color = (r, g, b)
            else:
                self.color = BALL_COLOR
        except (ValueError, TypeError):
            self.color = BALL_COLOR

    def launch(self, speed: float = 8):
        angle = random.uniform(-math.pi/4, math.pi/4)
        self.speed_x = math.sin(angle) * speed
        self.speed_y = -math.cos(angle) * speed
        # Ensure minimum speed
        self.normalize_speed()

    def normalize_speed(self):
        # Ensure minimum speed and cap maximum speed
        speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
        if speed < self.min_speed:
            scale = self.min_speed / speed if speed > 0 else 1
            self.speed_x *= scale
            self.speed_y *= scale
        elif speed > self.max_speed:
            scale = self.max_speed / speed
            self.speed_x *= scale
            self.speed_y *= scale

    def update(self, dt: float, slow_motion: bool = False):
        # Store previous position for collision resolution
        prev_x = self.x
        prev_y = self.y

        # Update position
        speed_multiplier = 0.3 if slow_motion else 1.0
        self.x += self.speed_x * speed_multiplier
        self.y += self.speed_y * speed_multiplier

        # Update laser cooldown
        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1

        # Calculate rotation based on movement
        if abs(self.speed_x) > 0.1 or abs(self.speed_y) > 0.1:
            self.rotation = math.degrees(math.atan2(self.speed_y, self.speed_x))

        # Calculate stretch based on speed
        speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
        self.stretch = 1.0 + min(0.5, speed / 20)

        # Ensure minimum speed
        self.normalize_speed()

        return prev_x, prev_y

    def draw(self, surface, shake_x: float = 0, shake_y: float = 0):
        # Draw glow with adjusted size
        for i in range(6, 0, -1):
            alpha = int(20 * i)
            glow_surf = pygame.Surface((self.radius*3, self.radius*3), pygame.SRCALPHA)
            r, g, b = self.color
            glow_color = (r, g, b, alpha)
            pygame.draw.circle(glow_surf, glow_color,
                             (int(self.radius*1.5), int(self.radius*1.5)),
                             int(self.radius * self.scale * self.stretch + i*2))
            surface.blit(glow_surf,
                        (self.x - self.radius*1.5 + shake_x,
                         self.y - self.radius*1.5 + shake_y),
                        special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main ball
        ball_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(ball_surf, self.color,
                         (self.radius, self.radius), self.radius)

        # Draw laser indicator if active
        if self.is_laser:
            pygame.draw.circle(ball_surf, (255, 50, 50),
                             (self.radius, self.radius), self.radius//3)

        if self.rotation != 0:
            ball_surf = pygame.transform.rotate(ball_surf, self.rotation)
        if self.scale * self.stretch != 1.0:
            new_size = (int(self.radius*2*self.scale*self.stretch),
                       int(self.radius*2*self.scale*self.stretch))
            ball_surf = pygame.transform.scale(ball_surf, new_size)

        surface.blit(ball_surf,
                    (self.x - ball_surf.get_width()//2 + shake_x,
                     self.y - ball_surf.get_height()//2 + shake_y))

class BreakoutGame:
    def __init__(self):
        self.screen_shake = ScreenShake()
        self.particle_system = ParticleSystem()
        self.tween_manager = TweenManager()
        self.paddle = Paddle()
        self.blocks = []
        self.powerups = []
        self.game_state = GameState.MENU
        self.score = 0
        self.lives = 3
        self.level = 1
        self.slow_motion = False
        self.slow_motion_timer = 0
        self.wide_paddle_timer = 0
        self.laser_timer = 0
        self.stars = [create_star() for _ in range(STAR_COUNT)]
        self.font = pygame.font.Font(None, int(48 * SCALE_FACTOR))
        self.small_font = pygame.font.Font(None, int(32 * SCALE_FACTOR))
        
        # Initialize folk music
        self.folk_music = generate_folk_melody()
        self.folk_music.set_volume(0.3)  # Set initial volume
        self.setup_level()

    def start_music(self):
        """Start playing the folk music"""
        self.folk_music.play(-1)  # -1 means loop indefinitely

    def stop_music(self):
        """Stop the folk music"""
        self.folk_music.stop()

    def setup_level(self):
        self.blocks.clear()
        self.powerups.clear()
        
        # Calculate block grid
        total_width = BLOCK_COLS * (BLOCK_WIDTH + BLOCK_PADDING) - BLOCK_PADDING
        start_x = (WINDOW_WIDTH - total_width) // 2
        start_y = int(100 * SCALE_FACTOR)

        # Create blocks
        for row in range(BLOCK_ROWS):
            for col in range(BLOCK_COLS):
                x = start_x + col * (BLOCK_WIDTH + BLOCK_PADDING)
                y = start_y + row * (BLOCK_HEIGHT + BLOCK_PADDING)
                
                # Determine block type based on row and level
                if row == 0 and random.random() < 0.3:
                    block_type = BlockType.INDESTRUCTIBLE
                elif row < 2:
                    block_type = BlockType.HARD
                elif row < 4 and random.random() < 0.2:
                    block_type = BlockType.POWERUP
                else:
                    block_type = BlockType.NORMAL
                
                self.blocks.append(Block(x, y, block_type, row))

        # Reset ball position - ensure it's below all blocks
        lowest_block_y = max(block.y + block.height for block in self.blocks) if self.blocks else start_y
        self.balls = [Ball(WINDOW_WIDTH//2, lowest_block_y + 100)]  # Position ball well below blocks
        self.balls[0].launch()

    def handle_powerup(self, powerup_type: PowerUpType):
        if powerup_type == PowerUpType.WIDE_PADDLE:
            self.wide_paddle_timer = 600  # 10 seconds
            self.paddle.width = PADDLE_WIDTH * 1.5
        elif powerup_type == PowerUpType.MULTI_BALL:
            for ball in self.balls[:]:  # Copy list to avoid modification during iteration
                angle = random.uniform(-math.pi/3, math.pi/3)
                speed = math.sqrt(ball.speed_x**2 + ball.speed_y**2)
                new_ball = Ball(ball.x, ball.y)
                new_ball.speed_x = math.sin(angle) * speed
                new_ball.speed_y = math.cos(angle) * speed
                new_ball.color = ball.color
                self.balls.append(new_ball)
        elif powerup_type == PowerUpType.SLOW_BALL:
            self.slow_motion_timer = 300  # 5 seconds
        elif powerup_type == PowerUpType.EXTRA_LIFE:
            self.lives = min(self.lives + 1, 5)
        elif powerup_type == PowerUpType.LASER:
            self.laser_timer = 600  # 10 seconds
            for ball in self.balls:
                ball.is_laser = True

    def update(self, dt: float):
        if self.game_state == GameState.MENU:
            self.stop_music()
        elif self.game_state == GameState.PLAYING and not pygame.mixer.get_busy():
            self.start_music()
        elif self.game_state in [GameState.GAME_OVER, GameState.LEVEL_COMPLETE]:
            self.stop_music()

        if self.game_state != GameState.PLAYING:
            return

        # Update paddle
        self.paddle.update(self.balls[0].x, self.balls[0].y)

        # Update power-up timers
        if self.wide_paddle_timer > 0:
            self.wide_paddle_timer -= 1
            if self.wide_paddle_timer == 0:
                self.paddle.width = PADDLE_WIDTH

        if self.laser_timer > 0:
            self.laser_timer -= 1
            if self.laser_timer == 0:
                for ball in self.balls:
                    ball.is_laser = False

        if self.slow_motion_timer > 0:
            self.slow_motion_timer -= 1
            self.slow_motion = True
        else:
            self.slow_motion = False

        # Update balls
        for ball in self.balls[:]:  # Copy list to avoid modification during iteration
            prev_x, prev_y = ball.update(dt, self.slow_motion)

            # Check wall collisions with improved response
            if ball.x <= ball.radius:
                ball.x = ball.radius
                ball.speed_x = abs(ball.speed_x)
                self.screen_shake.shake(0.3)
                self.particle_system.emit_shatter(ball.x, ball.y, ball.color)
                play_bounce_sound()
            elif ball.x >= WINDOW_WIDTH - ball.radius:
                ball.x = WINDOW_WIDTH - ball.radius
                ball.speed_x = -abs(ball.speed_x)
                self.screen_shake.shake(0.3)
                self.particle_system.emit_shatter(ball.x, ball.y, ball.color)
                play_bounce_sound()

            if ball.y <= ball.radius:
                ball.y = ball.radius
                ball.speed_y = abs(ball.speed_y)
                self.screen_shake.shake(0.3)
                self.particle_system.emit_shatter(ball.x, ball.y, ball.color)
                play_bounce_sound()

            # Check paddle collision with improved response
            if (ball.y + ball.radius >= self.paddle.y and 
                ball.y - ball.radius <= self.paddle.y + self.paddle.height and
                ball.x + ball.radius >= self.paddle.x and 
                ball.x - ball.radius <= self.paddle.x + self.paddle.width):
                
                # Calculate bounce angle based on where the ball hits the paddle
                relative_intersect = (ball.x - (self.paddle.x + self.paddle.width/2)) / (self.paddle.width/2)
                bounce_angle = relative_intersect * math.pi/3
                
                # Ensure minimum vertical speed after paddle hit
                speed = max(ball.min_speed, math.sqrt(ball.speed_x**2 + ball.speed_y**2))
                ball.speed_x = math.sin(bounce_angle) * speed
                ball.speed_y = -abs(math.cos(bounce_angle) * speed)  # Always bounce upward
                
                # Move ball above paddle to prevent sticking
                ball.y = self.paddle.y - ball.radius - 1  # Added small offset
                
                # Change ball color on paddle hit with valid RGB values
                new_color = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
                ball.set_color(new_color)
                
                self.screen_shake.shake(0.4)
                self.particle_system.emit_confetti(ball.x, ball.y)
                self.paddle.scale = 1.2
                self.tween_manager.add(Tween(1.2, 1.0, 0.2, easing_func=ease_out_elastic))
                play_bounce_sound()

            # Check if ball is lost
            if ball.y > WINDOW_HEIGHT + ball.radius:
                self.balls.remove(ball)
                if not self.balls:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_state = GameState.GAME_OVER
                    else:
                        # Reset ball position when losing a life
                        lowest_block_y = max(block.y + block.height for block in self.blocks) if self.blocks else 100
                        self.balls = [Ball(WINDOW_WIDTH//2, lowest_block_y + 100)]
                        self.balls[0].launch()

            # Check block collisions with improved response
            for block in self.blocks[:]:  # Copy list to avoid modification during iteration
                # More precise collision detection
                ball_rect = pygame.Rect(ball.x - ball.radius, ball.y - ball.radius,
                                      ball.radius * 2, ball.radius * 2)
                block_rect = pygame.Rect(block.x, block.y, block.width, block.height)
                
                if ball_rect.colliderect(block_rect):
                    # Calculate overlap in each direction
                    overlap_x = min(ball.x + ball.radius - block.x, block.x + block.width - (ball.x - ball.radius))
                    overlap_y = min(ball.y + ball.radius - block.y, block.y + block.height - (ball.y - ball.radius))
                    
                    if block.block_type == BlockType.INDESTRUCTIBLE:
                        # Bounce off indestructible blocks
                        if overlap_x < overlap_y:
                            ball.speed_x = -ball.speed_x
                            ball.x += -overlap_x if ball.speed_x < 0 else overlap_x
                        else:
                            ball.speed_y = -ball.speed_y
                            ball.y += -overlap_y if ball.speed_y < 0 else overlap_y
                        self.screen_shake.shake(0.2)
                        play_bounce_sound()
                    elif ball.is_laser:
                        # Laser destroys blocks instantly
                        powerup = block.hit()
                        if block.is_destroyed():
                            self.handle_powerup(powerup)
                            self.score += 100
                            self.blocks.remove(block)
                            block_color = BLOCK_COLORS[block.block_type][0]
                            self.particle_system.emit_shatter(block.x + block.width/2,
                                                            block.y + block.height/2,
                                                            block_color)
                    else:
                        # Normal block collision
                        if overlap_x < overlap_y:
                            ball.speed_x = -ball.speed_x
                            ball.x += -overlap_x if ball.speed_x < 0 else overlap_x
                        else:
                            ball.speed_y = -ball.speed_y
                            ball.y += -overlap_y if ball.speed_y < 0 else overlap_y
                        
                        powerup = block.hit()
                        if block.is_destroyed():
                            self.handle_powerup(powerup)
                            self.score += 100
                            self.blocks.remove(block)
                            block_color = BLOCK_COLORS[block.block_type][0]
                            self.particle_system.emit_shatter(block.x + block.width/2,
                                                            block.y + block.height/2,
                                                            block_color)
                            # Change ball color on block destruction
                            ball.set_color(block_color)
                        else:
                            self.screen_shake.shake(0.2)
                            block_color = BLOCK_COLORS[block.block_type][0]
                            self.particle_system.emit(block.x + block.width/2,
                                                    block.y + block.height/2,
                                                    10, block_color)
                        play_bounce_sound()
                        
                        # Ensure minimum speed after collision
                        ball.normalize_speed()

        # Update power-ups
        for powerup in self.powerups[:]:  # Copy list to avoid modification during iteration
            if not powerup.update():
                self.powerups.remove(powerup)
                continue

            # Check paddle collision
            if (powerup.y + powerup.height >= self.paddle.y and 
                powerup.y <= self.paddle.y + self.paddle.height and
                powerup.x + powerup.width >= self.paddle.x and 
                powerup.x <= self.paddle.x + self.paddle.width):
                self.handle_powerup(powerup.powerup_type)
                self.powerups.remove(powerup)
                self.particle_system.emit_confetti(powerup.x, powerup.y)
                play_impact_sound()

        # Check level completion
        if not self.blocks:
            self.level += 1
            if self.level > 3:  # Game completed
                self.game_state = GameState.MENU
            else:
                self.game_state = GameState.LEVEL_COMPLETE
                self.setup_level()

        # Update effects
        self.screen_shake.update()
        self.particle_system.update()
        self.tween_manager.update(dt)
        update_stars(self.stars)

    def draw(self, surface):
        # Apply screen shake
        shake_x = self.screen_shake.offset_x
        shake_y = self.screen_shake.offset_y
        
        # Draw background
        surface.fill(BACKGROUND_COLOR)
        draw_stars(surface, self.stars)
        
        # Draw particles
        self.particle_system.draw(surface)
        
        # Draw blocks
        for block in self.blocks:
            block.draw(surface)
        
        # Draw power-ups
        for powerup in self.powerups:
            powerup.draw(surface)
        
        # Draw balls
        for ball in self.balls:
            ball.draw(surface, shake_x, shake_y)
        
        # Draw paddle
        self.paddle.draw(surface)
        
        # Draw UI
        if self.game_state == GameState.PLAYING:
            # Draw score
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            surface.blit(score_text, (20, 20))
            
            # Draw lives
            lives_text = self.font.render(f"Lives: {self.lives}", True, (255, 255, 255))
            surface.blit(lives_text, (WINDOW_WIDTH - lives_text.get_width() - 20, 20))
            
            # Draw level
            level_text = self.font.render(f"Level: {self.level}", True, (255, 255, 255))
            surface.blit(level_text, (WINDOW_WIDTH//2 - level_text.get_width()//2, 20))
            
            # Draw power-up indicators
            if self.wide_paddle_timer > 0:
                pygame.draw.rect(surface, POWERUP_COLORS[PowerUpType.WIDE_PADDLE],
                               (20, 80, 20, 20))
            if self.laser_timer > 0:
                pygame.draw.rect(surface, POWERUP_COLORS[PowerUpType.LASER],
                               (50, 80, 20, 20))
            if self.slow_motion_timer > 0:
                pygame.draw.rect(surface, POWERUP_COLORS[PowerUpType.SLOW_BALL],
                               (80, 80, 20, 20))
        
        elif self.game_state == GameState.MENU:
            # Draw menu
            title = self.font.render("SPACE BREAKOUT", True, (255, 255, 255))
            start_text = self.small_font.render("Press SPACE to Start", True, (200, 200, 200))
            surface.blit(title, (WINDOW_WIDTH//2 - title.get_width()//2, WINDOW_HEIGHT//3))
            surface.blit(start_text, (WINDOW_WIDTH//2 - start_text.get_width()//2, WINDOW_HEIGHT//2))
        
        elif self.game_state == GameState.GAME_OVER:
            # Draw game over
            game_over = self.font.render("GAME OVER", True, (255, 50, 50))
            score_text = self.small_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
            restart_text = self.small_font.render("Press SPACE to Restart", True, (200, 200, 200))
            surface.blit(game_over, (WINDOW_WIDTH//2 - game_over.get_width()//2, WINDOW_HEIGHT//3))
            surface.blit(score_text, (WINDOW_WIDTH//2 - score_text.get_width()//2, WINDOW_HEIGHT//2))
            surface.blit(restart_text, (WINDOW_WIDTH//2 - restart_text.get_width()//2, WINDOW_HEIGHT*2//3))
        
        elif self.game_state == GameState.LEVEL_COMPLETE:
            # Draw level complete
            level_complete = self.font.render(f"LEVEL {self.level-1} COMPLETE!", True, (50, 255, 50))
            continue_text = self.small_font.render("Press SPACE to Continue", True, (200, 200, 200))
            surface.blit(level_complete, (WINDOW_WIDTH//2 - level_complete.get_width()//2, WINDOW_HEIGHT//3))
            surface.blit(continue_text, (WINDOW_WIDTH//2 - continue_text.get_width()//2, WINDOW_HEIGHT//2))

def create_star():
    return {
        'x': random.randint(0, WINDOW_WIDTH),
        'y': random.randint(0, WINDOW_HEIGHT),
        'speed': random.uniform(*STAR_SPEED_RANGE),
        'size': random.randint(2, 4)  # Bigger stars
    }

def update_stars(stars):
    for star in stars:
        star['y'] += star['speed']
        if star['y'] > WINDOW_HEIGHT:
            star['x'] = random.randint(0, WINDOW_WIDTH)
            star['y'] = 0
            star['speed'] = random.uniform(*STAR_SPEED_RANGE)
            star['size'] = random.randint(2, 4)

def draw_stars(surface, stars):
    for star in stars:
        pygame.draw.circle(surface, STAR_COLOR, (int(star['x']), int(star['y'])), star['size'])

def draw_ball_with_effects(surface, x, y, color, scale, rotation):
    # Draw glow
    for i in range(8, 0, -1):
        alpha = int(20 * i)
        glow_surf = pygame.Surface((BALL_RADIUS*4, BALL_RADIUS*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color[:3], alpha), 
                         (BALL_RADIUS*2, BALL_RADIUS*2), 
                         int(BALL_RADIUS * scale + i*3))
        surface.blit(glow_surf, (x-BALL_RADIUS*2, y-BALL_RADIUS*2), 
                    special_flags=pygame.BLEND_RGBA_ADD)

    # Draw main ball with rotation and scale
    ball_surf = pygame.Surface((BALL_RADIUS*2, BALL_RADIUS*2), pygame.SRCALPHA)
    pygame.draw.circle(ball_surf, color, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
    
    if rotation != 0:
        ball_surf = pygame.transform.rotate(ball_surf, rotation)
    if scale != 1.0:
        new_size = (int(BALL_RADIUS*2*scale), int(BALL_RADIUS*2*scale))
        ball_surf = pygame.transform.scale(ball_surf, new_size)
    
    surface.blit(ball_surf, (x - ball_surf.get_width()//2, y - ball_surf.get_height()//2))

def generate_tone(frequency, duration_ms, volume=0.2):
    sample_rate = 44100
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = np.int16(tone * volume * 32767)
    stereo_audio = np.column_stack((audio, audio))
    return pygame.sndarray.make_sound(stereo_audio)

def play_bounce_sound():
    freq = random.choice([330, 440, 550, 660, 880])
    s = generate_tone(freq, 100, 0.4)  # Longer sound
    s.play()

def play_impact_sound():
    freq = random.choice([220, 277, 330, 440])
    s = generate_tone(freq, 150, 0.5)  # Longer sound
    s.play()

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Space Breakout!")
    clock = pygame.time.Clock()
    
    game = BreakoutGame()
    
    while True:
        dt = 1/FPS  # Fixed time step
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    # Toggle fullscreen
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                elif event.key == pygame.K_SPACE:
                    if game.game_state == GameState.MENU:
                        game.game_state = GameState.PLAYING
                        game.start_music()  # Start music when game starts
                    elif game.game_state == GameState.GAME_OVER:
                        game = BreakoutGame()  # Reset game
                    elif game.game_state == GameState.LEVEL_COMPLETE:
                        game.game_state = GameState.PLAYING
                        game.start_music()  # Resume music when continuing

        game.update(dt)
        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main() 
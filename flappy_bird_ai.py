import cv2
import mediapipe as mp
import pygame
import sys
import random
import numpy as np
import torch
from flappy_bird_rl import FlappyBirdAgent, AIPersonality, create_ai_personalities

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

pygame.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('AI-Enhanced Flappy Bird')
clock = pygame.time.Clock()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)

def load_image(path, size=None):
    try:
        img = pygame.image.load(path).convert_alpha()
        if size:
            img = pygame.transform.scale(img, size)
        return img
    except:
        # Create colored rectangle if image not found
        surf = pygame.Surface(size if size else (30, 30))
        surf.fill(blue)
        return surf

def create_ai_emoji_fallback(size=(30, 30)):
    """Create a fallback AI emoji if aiemoji.png is not found"""
    surf = pygame.Surface(size)
    surf.fill(green)
    # Add "AI" text on the green surface
    font = pygame.font.SysFont(None, 16)
    text = font.render("AI", True, black)
    text_rect = text.get_rect(center=(size[0]//2, size[1]//2))
    surf.blit(text, text_rect)
    return surf

# Load resources (with fallbacks)
try:
    background_paths = ['bg1.png', 'bg2.png', 'bg3.png']
    backgrounds = [load_image(p, (width, height)) for p in background_paths]
except:
    # Create gradient backgrounds as fallback
    backgrounds = []
    for i, color in enumerate([(135, 206, 235), (255, 140, 0), (72, 61, 139)]):
        surf = pygame.Surface((width, height))
        surf.fill(color)
        backgrounds.append(surf)

current_bg = backgrounds[0]
bird_image = load_image('bird.png', (30, 30))

# Load AI emoji
try:
    ai_emoji = load_image('aiemoji.png', (30, 30))
    print("AI emoji loaded successfully!")
except:
    # Create fallback AI emoji
    ai_emoji = create_ai_emoji_fallback((30, 30))
    print("AI emoji not found, using fallback green emoji with 'AI' text")

# Load other emojis with proper sizes like in original
death_emojis = [load_image(p, (60, 60)) for p in ['emoji1.png', 'emoji2.png', 'emoji3.png']]
highscore_emojis = [load_image(p, (80, 80)) for p in ['hs1.png', 'hs2.png']]
milestone_emojis = [load_image(p, (50, 50)) for p in ['ms1.png', 'ms2.png']]

# Initialize sounds
try:
    pygame.mixer.init()
    beep_sound = pygame.mixer.Sound('crash.mp3')
    pass_sound = pygame.mixer.Sound('pass.mp3')
    wow_sound = pygame.mixer.Sound('wow.mp3')
except:
    beep_sound = None
    pass_sound = None
    wow_sound = None

current_emoji = bird_image
current_ai_emoji = ai_emoji  

class GameAI:
    def __init__(self):
        self.agent = None
        self.ai_personalities = create_ai_personalities()
        self.current_personality = None
        self.ai_mode = "OFF"  
        self.ai_bird_x = 100
        self.ai_bird_y = height // 2
        self.ai_bird_speed_y = 0
        self.ai_score = 0
        self.ai_passed_pipe = False  
        self.load_agent()
        
    def load_agent(self):
        """Load the trained AI agent"""
        try:
            self.agent = FlappyBirdAgent()
            self.agent.load("flappy_bird_agent.pth")
            print("AI Agent loaded successfully!")
        except FileNotFoundError:
            print("No trained AI agent found. AI features disabled.")
            self.agent = None
    
    def get_game_state(self, bird_y, bird_speed_y, pipe_x, pipe_y, pipe_gap=200):
        """Convert game state to AI input format"""
        bird_y_norm = bird_y / height
        bird_speed_norm = (bird_speed_y + 10) / 20
        pipe_x_norm = pipe_x / width
        pipe_y_norm = pipe_y / height
        
        distance_to_pipe_top = (pipe_y - pipe_gap//2 - bird_y) / height
        distance_to_pipe_bottom = (pipe_y + pipe_gap//2 - bird_y) / height
        distance_to_pipe_x = (pipe_x - 100) / width  # Assuming bird_x = 100
        
        return np.array([
            bird_y_norm,
            bird_speed_norm,
            distance_to_pipe_x,
            distance_to_pipe_top,
            distance_to_pipe_bottom,
            pipe_x_norm,
            pipe_y_norm
        ], dtype=np.float32)
    
    def get_ai_action(self, bird_y, bird_speed_y, pipe_x, pipe_y):
        """Get AI action for current game state"""
        if not self.agent:
            return 0
        
        state = self.get_game_state(bird_y, bird_speed_y, pipe_x, pipe_y)
        base_action = self.agent.act(state, training=False)
        
        # Apply personality modification if active
        if self.current_personality:
            return self.current_personality.modify_action(base_action, state, self.agent)
        
        return base_action
    
    def update_ai_bird(self, pipe_x, pipe_y, pipe_width):
        """Update AI bird position"""
        if not self.agent:
            return
        
        # Get AI action
        ai_action = self.get_ai_action(self.ai_bird_y, self.ai_bird_speed_y, pipe_x, pipe_y)
        
        # Apply action
        if ai_action == 1:
            self.ai_bird_speed_y = -5  # Use same jump speed as player
        
        # Apply physics
        self.ai_bird_speed_y += 0.5  # gravity
        self.ai_bird_y += self.ai_bird_speed_y
        
        # Check if AI bird passed pipe for scoring
        if pipe_x < self.ai_bird_x and pipe_x + pipe_width >= self.ai_bird_x and not self.ai_passed_pipe:
            self.ai_score += 1
            self.ai_passed_pipe = True
            play_sound(pass_sound)
            
            # Update AI emoji for celebrations
            global current_ai_emoji
            if self.ai_score % 5 == 0:
                current_ai_emoji = random.choice(milestone_emojis) if milestone_emojis else ai_emoji
            else:
                current_ai_emoji = ai_emoji
        
        # Reset pipe passing flag when new pipe comes
        if pipe_x > self.ai_bird_x + pipe_width:
            self.ai_passed_pipe = False
        
        # Check AI bird collision
        if (self.ai_bird_y <= 0 or self.ai_bird_y >= height - 30):
            self.reset_ai_bird()
    
    def check_ai_collision(self, pipe_x, pipe_y, pipe_gap, pipe_width, pipe_height):
        """Check if AI bird collides with pipes"""
        ai_bird_rect = pygame.Rect(self.ai_bird_x, self.ai_bird_y, 30, 30)
        pipe_top_rect = pygame.Rect(pipe_x, pipe_y - pipe_gap//2 - pipe_height, pipe_width, pipe_height)
        pipe_bottom_rect = pygame.Rect(pipe_x, pipe_y + pipe_gap//2, pipe_width, pipe_height)
        
        if ai_bird_rect.colliderect(pipe_top_rect) or ai_bird_rect.colliderect(pipe_bottom_rect):
            self.reset_ai_bird()
            return True
        return False
    
    def reset_ai_bird(self):
        """Reset AI bird position"""
        self.ai_bird_y = height // 2
        self.ai_bird_speed_y = 0
        self.ai_score = 0
        self.ai_passed_pipe = False
        global current_ai_emoji
        current_ai_emoji = ai_emoji  # Reset to normal AI emoji
    
    def draw_ai_bird(self, screen):
        """Draw the AI bird with emoji"""
        if self.ai_mode in ["COMPETE", "DEMO"]:
            # Draw AI bird with emoji
            screen.blit(current_ai_emoji, (self.ai_bird_x, self.ai_bird_y))
            
            # Draw AI label
            font = pygame.font.SysFont(None, 20)
            ai_text = font.render("AI", True, green)
            screen.blit(ai_text, (self.ai_bird_x + 35, self.ai_bird_y + 5))
    
    def draw_ai_assistance(self, screen, bird_y, bird_speed_y, pipe_x, pipe_y):
        """Draw AI assistance indicators"""
        if self.ai_mode == "ASSIST" and self.agent:
            # Get AI recommendation
            ai_action = self.get_ai_action(bird_y, bird_speed_y, pipe_x, pipe_y)
            
            # Draw recommendation with AI emoji
            if ai_action == 1:
                pygame.draw.circle(screen, green, (50, 50), 25)
                screen.blit(ai_emoji, (35, 35))
                font = pygame.font.SysFont(None, 24)
                text = font.render("JUMP!", True, black)
                screen.blit(text, (10, 85))
            else:
                pygame.draw.circle(screen, red, (50, 50), 25)
                screen.blit(ai_emoji, (35, 35))
                font = pygame.font.SysFont(None, 24)
                text = font.render("WAIT", True, black)
                screen.blit(text, (10, 85))

# Initialize AI
game_ai = GameAI()

# Game state
class GameState:
    def __init__(self):
        self.bird_x = 100
        self.bird_y = height // 2
        self.bird_speed_y = 0
        self.pipe_x = width
        self.pipe_y = random.randint(200, height - 200)
        self.pipe_gap = 200
        self.pipe_width = 60
        self.pipe_height = 300
        self.pipe_speed = 3  # Initial pipe speed
        self.score = 0
        self.highest_score = self.load_highest_score()
        self.gravity = 0.5
        self.jump_speed = -5  # Same as original
        self.fall_speed = 5   # Same as original
        self.game_over = False
        self.passed_pipe = False  # Track if player passed pipe for scoring
    
    def load_highest_score(self):
        try:
            return int(open('highest_score.txt').read())
        except:
            return 0
    
    def save_highest_score(self, score):
        open('highest_score.txt', 'w').write(str(score))
    
    def reset(self):
        self.bird_y = height // 2
        self.bird_speed_y = 0
        self.pipe_x = width
        self.pipe_y = random.randint(200, height - 200)
        self.pipe_speed = 3  # Reset to initial speed
        self.score = 0
        self.passed_pipe = False
        self.game_over = False
        game_ai.reset_ai_bird()
        global current_bg, current_emoji, current_ai_emoji
        current_bg = backgrounds[0]  # Reset background
        current_emoji = bird_image    # Reset emoji
        current_ai_emoji = ai_emoji   # Reset AI emoji

# Gesture detection functions
def is_thumbs_up(lm):
    tip = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
    mcp = lm.landmark[mp_hands.HandLandmark.THUMB_MCP]
    idx = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return tip.y < mcp.y and tip.y < idx.y

def is_thumbs_down(lm):
    tip = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
    mcp = lm.landmark[mp_hands.HandLandmark.THUMB_MCP]
    idx = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return tip.y > mcp.y and tip.y > idx.y

def update_background(score):
    """Update background based on score like in original"""
    if score < 5:
        return backgrounds[0]
    elif score < 10:
        return backgrounds[1]
    else:
        return backgrounds[2]

def display_emoji(img, pos, frames=30):
    """Display emoji animation like in original"""
    for _ in range(frames):
        screen.blit(current_bg, (0, 0))
        screen.blit(img, pos)
        pygame.display.flip()
        clock.tick(60)

def play_sound(sound):
    """Play sound if available"""
    if sound:
        sound.play()

def show_score(screen, score, highest_score, ai_score=0):
    font = pygame.font.SysFont(None, 35)  # Same size as original
    text = font.render(f"Score: {score}  High: {highest_score}", True, black)
    screen.blit(text, (10, 10))
    
    if game_ai.ai_mode in ["COMPETE", "DEMO"]:
        ai_text = font.render(f"AI: {ai_score}", True, green)
        screen.blit(ai_text, (10, 50))
        # Add AI emoji next to AI score
        screen.blit(ai_emoji, (70, 55))

def show_ai_controls(screen):
    """Show AI mode controls"""
    font = pygame.font.SysFont(None, 24)
    controls = [
        "AI Controls:",
        "1 - AI Off",
        "2 - AI Assist",
        "3 - AI Compete",
        "4 - AI Demo",
        "5-9 - AI Personalities"
    ]
    
    for i, control in enumerate(controls):
        color = white if i == 0 else black
        text = font.render(control, True, color)
        screen.blit(text, (width - 200, 10 + i * 25))
    
    # Add AI emoji next to controls
    screen.blit(ai_emoji, (width - 230, 10))

def handle_ai_mode_keys(keys):
    """Handle AI mode switching"""
    if keys[pygame.K_1]:
        game_ai.ai_mode = "OFF"
        print("AI Mode: OFF")
    elif keys[pygame.K_2]:
        game_ai.ai_mode = "ASSIST"
        print("AI Mode: ASSIST")
    elif keys[pygame.K_3]:
        game_ai.ai_mode = "COMPETE"
        print("AI Mode: COMPETE")
    elif keys[pygame.K_4]:
        game_ai.ai_mode = "DEMO"
        print("AI Mode: DEMO")
    elif keys[pygame.K_5]:
        game_ai.current_personality = game_ai.ai_personalities[0]
        print(f"AI Personality: {game_ai.current_personality.name}")
    elif keys[pygame.K_6]:
        game_ai.current_personality = game_ai.ai_personalities[1]
        print(f"AI Personality: {game_ai.current_personality.name}")
    elif keys[pygame.K_7]:
        game_ai.current_personality = game_ai.ai_personalities[2]
        print(f"AI Personality: {game_ai.current_personality.name}")
    elif keys[pygame.K_8]:
        game_ai.current_personality = game_ai.ai_personalities[3]
        print(f"AI Personality: {game_ai.current_personality.name}")
    elif keys[pygame.K_9]:
        game_ai.current_personality = game_ai.ai_personalities[4]
        print(f"AI Personality: {game_ai.current_personality.name}")

def start_screen():
    """Enhanced start screen with AI options"""
    while True:
        screen.blit(current_bg, (0, 0))
        
        # Title
        title_font = pygame.font.SysFont(None, 48)
        title_text = title_font.render('AI-Enhanced Flappy Bird', True, black)
        screen.blit(title_text, (width//2 - 200, height//2 - 100))
        
        # Add AI emoji next to title
        screen.blit(ai_emoji, (width//2 + 220, height//2 - 95))
        
        # Instructions
        font = pygame.font.SysFont(None, 32)
        instructions = [
            'Click to Play',
            'Use gestures or keyboard',
            'Press 1-4 for AI modes',
            'Press 5-9 for AI personalities'
        ]
        
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, black)
            screen.blit(text, (width//2 - 150, height//2 - 20 + i * 35))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                return

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam - using keyboard controls only")
    cap = None

# Start screen
start_screen()

# Initialize game
game = GameState()

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not game.game_over:
                    game.bird_speed_y = game.jump_speed
    
    # Handle AI mode keys
    keys = pygame.key.get_pressed()
    handle_ai_mode_keys(keys)
    
    # Process webcam input for gestures
    gesture = None
    if cap:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
            
            # Process gesture
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    if is_thumbs_up(lm):
                        gesture = 'up'
                    elif is_thumbs_down(lm):
                        gesture = 'down'
    
    # Apply controls (only if not in DEMO mode or game over)
    if not game.game_over:
        if game_ai.ai_mode == "DEMO":
            # AI controls the player bird in demo mode
            ai_action = game_ai.get_ai_action(game.bird_y, game.bird_speed_y, game.pipe_x, game.pipe_y)
            if ai_action == 1:
                game.bird_speed_y = game.jump_speed
        else:
            # Human controls
            if gesture == 'up':
                game.bird_speed_y = game.jump_speed
            elif gesture == 'down':
                game.bird_speed_y = game.fall_speed
            else:
                game.bird_speed_y += game.gravity
        
        # Apply physics
        if game_ai.ai_mode != "DEMO":
            game.bird_speed_y += game.gravity
        game.bird_y += game.bird_speed_y
        
        # Update AI bird if competing
        if game_ai.ai_mode == "COMPETE":
            game_ai.update_ai_bird(game.pipe_x, game.pipe_y, game.pipe_width)
            # Check AI collision with pipes
            game_ai.check_ai_collision(game.pipe_x, game.pipe_y, game.pipe_gap, game.pipe_width, game.pipe_height)
        
        # Move pipes
        game.pipe_x -= game.pipe_speed
        
        # Check if player passed pipe for scoring
        if game.pipe_x < game.bird_x and game.pipe_x + game.pipe_width >= game.bird_x and not game.passed_pipe:
            game.score += 1
            game.passed_pipe = True
            
            # Update background based on score
            current_bg = update_background(game.score)
            
            # Handle milestone celebrations like in original
            if game.score % 5 == 0:
                game.pipe_speed += 0.5
                display_emoji(random.choice(milestone_emojis), (width//2-25, height//2-25))
            
            # Handle high score celebrations
            if game.score > game.highest_score:
                display_emoji(random.choice(highscore_emojis), (width//2-40, height//2-40))
                game.highest_score = game.score
                game.save_highest_score(game.score)
                play_sound(wow_sound)
            else:
                play_sound(pass_sound)
        
        # Reset pipe and generate new one
        if game.pipe_x < -game.pipe_width:
            game.pipe_x = width
            game.pipe_y = random.randint(200, height - 200)
            game.passed_pipe = False
        
        # Check collisions
        bird_rect = pygame.Rect(game.bird_x, game.bird_y, 30, 30)
        pipe_top_rect = pygame.Rect(game.pipe_x, game.pipe_y - game.pipe_gap//2 - game.pipe_height, game.pipe_width, game.pipe_height)
        pipe_bottom_rect = pygame.Rect(game.pipe_x, game.pipe_y + game.pipe_gap//2, game.pipe_width, game.pipe_height)
        
        if (game.bird_y <= 0 or game.bird_y >= height - 30 or
            bird_rect.colliderect(pipe_top_rect) or bird_rect.colliderect(pipe_bottom_rect)):
            game.game_over = True
            play_sound(beep_sound)
            
            # Display death emoji animation
            display_emoji(random.choice(death_emojis), (game.bird_x, game.bird_y))
            
            # Update emoji for game over state
            current_emoji = random.choice(death_emojis)
    
    # Draw everything
    screen.blit(current_bg, (0, 0))
    
    # Draw pipes
    pygame.draw.rect(screen, red, (game.pipe_x, game.pipe_y - game.pipe_gap//2 - game.pipe_height, game.pipe_width, game.pipe_height))
    pygame.draw.rect(screen, red, (game.pipe_x, game.pipe_y + game.pipe_gap//2, game.pipe_width, game.pipe_height))
    
    # Update and draw player bird emoji based on game state
    if not game.game_over:
        if game.score > game.highest_score:
            current_emoji = random.choice(highscore_emojis) if random.randint(1, 10) == 1 else bird_image
        elif game.score > 0 and game.score % 10 == 0:
            current_emoji = random.choice(milestone_emojis) if random.randint(1, 10) == 1 else bird_image
        else:
            current_emoji = bird_image
    
    screen.blit(current_emoji, (game.bird_x, game.bird_y))
    
    # Draw AI bird if competing
    game_ai.draw_ai_bird(screen)
    
    # Draw AI assistance if enabled
    if game_ai.ai_mode == "ASSIST":
        game_ai.draw_ai_assistance(screen, game.bird_y, game.bird_speed_y, game.pipe_x, game.pipe_y)
    
    # Draw scores
    show_score(screen, game.score, game.highest_score, game_ai.ai_score)
    
    # Draw AI controls
    show_ai_controls(screen)
    
    # Draw current AI mode
    font = pygame.font.SysFont(None, 32)
    mode_text = font.render(f"AI Mode: {game_ai.ai_mode}", True, blue)
    screen.blit(mode_text, (10, height - 40))
    
    if game_ai.current_personality:
        personality_text = font.render(f"Personality: {game_ai.current_personality.name}", True, blue)
        screen.blit(personality_text, (10, height - 70))
    
    # Handle game over
    if game.game_over:
        # Game over screen like in original
        while True:
            screen.blit(current_bg, (0, 0))
            game_over_font = pygame.font.SysFont(None, 40)
            game_over_text = game_over_font.render('Click to Replay', True, black)
            screen.blit(game_over_text, (width//2 - 120, height//2))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.MOUSEBUTTONDOWN:
                    game.reset()
                    break
            else:
                continue
            break
        
        if not running:
            break
    
    pygame.display.flip()
    clock.tick(60)

# Cleanup
if cap:
    cap.release()
    cv2.destroyAllWindows()
pygame.quit()
sys.exit()
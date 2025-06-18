import cv2
import mediapipe as mp
import pygame
import sys
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

pygame.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Gesture Controlled Flappy Bird')
clock = pygame.time.Clock()

white = (255, 255, 255)
black = (0, 0, 0)
red   = (255, 0, 0)

def load_image(path, size=None):
    img = pygame.image.load(path).convert_alpha()
    if size:
        img = pygame.transform.scale(img, size)
    return img

background_paths = ['bg1.jpg', 'bg2.jpg', 'bg3.jpg']
backgrounds = [load_image(p, (width, height)) for p in background_paths]
current_bg = backgrounds[0]

bird_image = load_image('bird.jpg', (30, 30))

death_emojis = [load_image(p, (60,60)) for p in ['emoji1.jpg','emoji2.jpg','emoji3.jpg']]
highscore_emojis = [load_image(p, (80,80)) for p in ['hs1.jpg','hs2.jpg']]
milestone_emojis = [load_image(p, (50,50)) for p in ['ms1.jpg','ms2.jpg']]

# Sounds
pygame.mixer.init()
beep_sound = pygame.mixer.Sound('crash.mp3')
pass_sound = pygame.mixer.Sound('pass.mp3')
wow_sound = pygame.mixer.Sound('wow.mp3')

# Game physics
bird_width, bird_height = 30, 30
bird_x, bird_y = 100, height // 2
bird_speed_y = 0
gravity = 0.5
jump_speed = -5
fall_speed = 5
pipe_initial_speed = 3

# Score display
def show_score(score, highest_score):
    text = pygame.font.SysFont(None, 35).render(f"Score: {score}  High: {highest_score}", True, black)
    screen.blit(text, (10,10))

# Gesture detection

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

# Persistence

def load_highest_score():
    try:
        return int(open('highest_score.txt').read())
    except:
        return 0

def save_highest_score(s):
    open('highest_score.txt','w').write(str(s))


def display_emoji(img, pos, frames=30):
    for _ in range(frames):
        screen.blit(current_bg, (0,0))
        screen.blit(img, pos)
        pygame.display.flip()
        clock.tick(60)


def update_background(score):
    if score < 10:
        return backgrounds[0]
    elif score < 20:
        return backgrounds[1]
    else:
        return backgrounds[2]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    sys.exit()


def start_screen():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Hand Tracking', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()
        screen.blit(current_bg, (0,0))
        screen.blit(pygame.font.SysFont(None,40).render('Click to Play', True, black), (width//2-100, height//2))
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                return

start_screen()

pipe_width, pipe_height, pipe_gap = 60, 300, 200
pipe_speed = pipe_initial_speed
score = 0
pipe_x = width
pipe_y = random.randint(pipe_gap, height - pipe_gap)
highest_score = load_highest_score()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Show webcam separately
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) == 27:
        break

    # Process gesture
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = None
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            if is_thumbs_up(lm):
                gesture = 'up'
            elif is_thumbs_down(lm):
                gesture = 'down'

    # Apply controls
    if gesture == 'up': bird_speed_y = jump_speed
    elif gesture == 'down': bird_speed_y = fall_speed
    else: bird_speed_y += gravity
    bird_y += bird_speed_y

    # Move pipes
    pipe_x -= pipe_speed
    if pipe_x < -pipe_width:
        pipe_x = width
        pipe_y = random.randint(pipe_gap, height - pipe_gap)
        score += 1
        current_bg = update_background(score)
        if score % 5 == 0:
            pipe_speed += 0.5
            display_emoji(random.choice(milestone_emojis), (width//2-25, height//2-25))
        if score > highest_score:
            display_emoji(random.choice(highscore_emojis), (width//2-40, height//2-40))
            highest_score = score
            save_highest_score(score)
        (wow_sound if score == highest_score else pass_sound).play()

    # Collision detection
    bird_rect = pygame.Rect(bird_x, bird_y, bird_width, bird_height)
    top_pipe = pygame.Rect(pipe_x, pipe_y-pipe_gap-pipe_height, pipe_width, pipe_height)
    bottom_pipe = pygame.Rect(pipe_x, pipe_y, pipe_width, pipe_height)
    collided = bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe) or bird_y<0 or bird_y>height-bird_height

    # Draw game
    screen.blit(current_bg, (0,0))
    pygame.draw.rect(screen, red, top_pipe)
    pygame.draw.rect(screen, red, bottom_pipe)
    screen.blit(bird_image, (bird_x, bird_y))
    show_score(score, highest_score)
    pygame.display.flip()
    clock.tick(60)

    # Handle collision
    if collided:
        beep_sound.play()
        display_emoji(random.choice(death_emojis), (bird_x, bird_y))
        # Reset state
        bird_y = height//2; bird_speed_y=0
        pipe_x = width; pipe_y = random.randint(pipe_gap, height-pipe_gap)
        pipe_speed = pipe_initial_speed; score=0
        # Replay prompt
        while True:
            screen.blit(current_bg,(0,0))
            screen.blit(pygame.font.SysFont(None,40).render('Click to Replay', True, black),(width//2-120, height//2))
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type==pygame.QUIT: cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()
                if e.type==pygame.MOUSEBUTTONDOWN: break
            else: continue
            break

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()

# 🐦 Gesture-Based Flappy Bird (AI-Enhanced)

An accessible version of Flappy Bird built using **Python**, designed for specially abled individuals under the **DRID initiative**. This version combines **real-time hand gesture control** with an **AI-powered agent** using **Reinforcement Learning** to assist, compete, or play on behalf of the user.

---

## 🎮 Features

- ✋ **Gesture Control** using **MediaPipe + OpenCV**
  - Thumbs up: Bird jumps
  - Thumbs down: Bird descends quickly

- 🧠 **AI Integration (DQN)**
  - **OFF** – Player controls the bird manually
  - **ASSIST** – AI gives live jump/wait suggestions
  - **COMPETE** – AI plays alongside the user
  - **DEMO** – AI plays the game autonomously

- 🧍 **AI Personalities**
  - Conservative, Aggressive, Balanced, Chaotic, and Cautious behavior styles

- 😊 **Emoji Feedback**
  - Milestone, high score, and death emojis for immersive feedback

- 🔊 **Sound Effects**
  - Realistic crash, pass, and wow sounds for key game events

- 📈 **Adaptive Difficulty**
  - Pipe speed and background change with score progress

- 🏆 **High Score Tracking**
  - Persistent high score saved across sessions

- 🛠️ **Fallback Handling**
  - Works even if webcam, emojis, or audio are missing

---

## 🚀 How to Run

### Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt

```
Or install manually:
```
pip install pygame opencv-python mediapipe torch matplotlib

```

then run the game 

```
python flappy_bird_ai.py

```

# EmotionLens

Real-time stress detector using your webcam. No wearables, no sensors — just AI and a camera.

Detects stress from facial movements and shows CALM / MILD STRESS / STRESSED live on screen.

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run**
```bash
python app.py
```

The model downloads automatically on first run (~2MB).

Press `S` to save a screenshot. Press `Q` to quit.

---

## How It Works

- MediaPipe maps 478 points on your face in real time
- Measures brow furrow, eye tightening, lip press, and chin tension
- Combines them into a stress score from 0 to 1
- Smooths the score over 15 frames for stability

---

## Tech Stack

- Python 3.10
- MediaPipe 0.10.35
- OpenCV
- NumPy

---

## Roadmap

- [x] Module 1 — Facial stress detection
- [ ] Module 2 — Keystroke dynamics
- [ ] Module 3 — Gaze estimation
- [ ] Module 4 — Combine all signals with fairness correction

---

Built by Amulya Krupa — [LinkedIn](https://www.linkedin.com/in/amulyakrupa/)

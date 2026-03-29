# Handsign

Real-time **ASL (American Sign Language) to English** translator. The browser captures webcam frames, streams them over a WebSocket to a **FastAPI** backend, which extracts **MediaPipe Holistic** landmarks, runs a **sequence LSTM** model, and returns predicted words with confidence. The UI shows live captions, builds a sentence, and keeps a short history.

## Quick start

From the project root (`handsign/`):

1. **Backend**

   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```

2. **Frontend** (new terminal)

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open the URL Vite prints (usually `http://localhost:5173`). Copy `.env.example` to `frontend/.env` if you want to override API/WebSocket URLs.

Without `backend/model/model.h5`, the backend runs in **demo mode** with mock predictions so you can test the full pipeline.

## Collect training data

1. Install the same stack as the backend plus **matplotlib** for training plots:

   ```bash
   pip install -r backend/requirements.txt matplotlib
   ```

2. From `training/`:

   ```bash
   cd training
   python collect_data.py
   ```

3. Follow the on-screen prompts. For each sign, the script records **30 frames** per sequence and saves `training/data/{sign}/{sequence}.npy` (shape `(30, 258)`). Press **Q** to quit early. A `labels.txt` is written listing signs that were collected.

## Train the model

```bash
cd training
python train.py
```

This loads sequences with `dataset_loader.py`, trains the LSTM, saves the best checkpoint to `backend/model/model.h5`, writes `backend/model/labels.txt`, and saves `training/training_curves.png`.

Install **matplotlib** if it is not already installed (`pip install matplotlib`).

## Evaluate

```bash
cd training
python evaluate.py
```

Prints a classification report, saves `training/confusion_matrix.png`, and lists the top confused sign pairs.

## Run backend and frontend

- **Backend**: `cd backend && python main.py` — serves HTTP on port **8000** (`/health`, `/labels`) and WebSocket **`/ws`**.
- **Frontend**: `cd frontend && npm run dev` — Vite dev server with a proxy so `/ws` and `/api` forward to `http://localhost:8000` (see `vite.config.js`).

Optional env vars (see `.env.example`):

- `VITE_WS_URL` — WebSocket URL (default in dev: same host via proxy).
- `VITE_API_URL` — REST base URL (empty in dev uses `/api` proxy).

## Project structure

| Path | Role |
|------|------|
| `backend/main.py` | FastAPI app, CORS, WebSocket JPEG → landmarks → inference → smoother |
| `backend/inference.py` | Keras `InferenceEngine`, rolling 30-frame buffer, demo mocks |
| `backend/mediapipe_utils.py` | Holistic helpers, 258-D keypoints |
| `backend/smoother.py` | `PredictionSmoother` for stable word output |
| `backend/model/` | `model.h5`, `labels.txt` (after training) |
| `training/collect_data.py` | Webcam recording to `.npy` sequences |
| `training/train.py` | LSTM training and checkpoint export |
| `training/evaluate.py` | Metrics and confusion matrix |
| `training/dataset_loader.py` | Load `.npy` files, stratified splits |
| `training/data/` | `{sign}/{seq}.npy` training files |
| `frontend/` | React + Vite UI, `react-webcam`, no UI framework |

## Supported signs (default vocabulary)

The default label list (24 words) is:

hello, thanks, yes, no, please, sorry, help, good, bad, more, stop, love, what, where, who, how, name, friend, eat, drink, water, home, work, learn.

After training, labels come from `backend/model/labels.txt` (one per line).

## How to add new signs

1. Add the new sign name to the `SIGNS` list in `training/collect_data.py` (and optionally to `DEFAULT_LABELS` in `backend/inference.py` for demo mode before retraining).
2. Run `collect_data.py` to record sequences under `training/data/{new_sign}/`.
3. Retrain with `train.py` so `model.h5` and `labels.txt` include the new class.
4. Restart the backend so it loads the updated model and labels.

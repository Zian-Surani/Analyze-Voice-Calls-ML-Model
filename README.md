# 🎙️ Analyze Voice Calls - ML Model with RL & Sentiment Feedback

A voice-based AI assistant that explains government schemes to farmers, listens to their replies, and **learns how to speak better** through reinforcement learning. This project combines **speech-to-text**, **sentiment analysis**, **text-to-speech**, and **reward-based learning** to simulate a real-world rural advisory assistant.

---

## 📌 Key Features

| Feature | Description |
|--------|-------------|
| 🧠 Reinforcement Learning | Improves messaging using feedback from farmer responses |
| 🗣️ Text-to-Speech (TTS) | Talks to farmers using `pyttsx3` |
| 🧏 Speech-to-Text (STT) | Uses `OpenAI Whisper` to understand voice replies |
| 💬 Sentiment Analysis | Judges tone of voice to reward/penalize the message |
| 📊 Learning Loop | Stores performance, adjusts future prompts |
| 💾 Q-table Storage | Saves agent state using `pickle` for persistent learning |

---

## 🌾 Real-World Use Case

> “Imagine an AI calling a farmer and explaining a solar subsidy.  
> The farmer responds, unsure.  
> The AI understands the confusion — and next time, simplifies the message.”  

This project is a **foundation for voice-based rural assistants** — ideal for agriculture, helplines, and public outreach.

---

## 🚀 Getting Started

### 🧩 Prerequisites

Make sure you have Python 3.8+ and install the following:

```bash
pip install -r requirements.txt
```
Or install manually:


``` bash
pip install openai-whisper pyttsx3 textblob torch torchaudio stable-baselines3
```
Note: Whisper may require FFmpeg and compatible CUDA drivers.

##📂 Folder Structure
```
├── agent.py           # RL logic and Q-table
├── speech.py          # Handles TTS and STT
├── sentiment.py       # Analyzes farmer response tone
├── main.py            # Main loop for training the agent
├── utils/             # Helpers for saving/loading models
├── recordings/        # (Optional) Voice logs or audio samples
└── q_table.pkl        # Saved Q-table for learning history
```

## 🧪 How It Works
The assistant chooses a message to deliver.

It speaks using TTS (pyttsx3).

Farmer response is recorded or simulated and passed through Whisper STT.

Sentiment is analyzed (positive/neutral/negative).

A reward is calculated and Q-table updated.

The loop continues — improving the assistant's strategy.


## 🔄 Advanced Options
Enhancement	How to Activate
🧠 True RL Agent	Use Stable-Baselines3 instead of Q-table
🎧 Real Audio	Replace simulated replies with .wav recordings
🎤 Whisper STT	Use actual Whisper transcription for input
📈 Metrics	Log accuracy, sentiment trends, engagement

## 🛠️ Future Ideas
Replace synthetic voices with real farmer voice samples

Use GPT-4 to generate clearer messages dynamically

Integrate with IVR or Twilio for real-time call flow

Add translation for regional languages

## 👨‍🔬 Authors
Zian Rajeshkumar Surani
@Zian-Surani | AI for Good | SRM IST Trichy

## 📜 License
This project is licensed under the MIT License.

## 💡 Inspiration
Inspired by the vision of using AI for digital inclusion in agriculture, helping rural populations access schemes, services, and support through intelligent voice systems.

“Technology should talk to people — not the other way around.”

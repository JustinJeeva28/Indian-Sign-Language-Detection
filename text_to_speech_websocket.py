import os
import json
import asyncio
import websockets
import base64
import time
import threading
from dotenv import load_dotenv
import msvcrt  # Windows-only for real-time key capture
import pyaudio
from pydub import AudioSegment
import io

# Load the API key from the .env file
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_989f85658385d98264758b44568d4cfc670f1690d4f66570")
voice_id = 'Xb7hH8MSUJpSbSDYk0k2'
model_id = 'eleven_flash_v2_5'

class RealTimeInput:
    def __init__(self, pause_time=1.0):
        self.text = ""
        self.last_time = time.time()
        self.pause_time = pause_time
        self.lock = threading.Lock()
        self.running = True

    def start(self):
        threading.Thread(target=self._capture, daemon=True).start()

    def _capture(self):
        print("Type your text. Audio will be generated after a short pause. Press ESC to exit.")
        while self.running:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char == '\x1b':  # ESC to exit
                    self.running = False
                    break
                elif char == '\r':  # Enter
                    print()
                elif char == '\b':  # Backspace
                    with self.lock:
                        self.text = self.text[:-1]
                    print("\b \b", end="", flush=True)
                else:
                    with self.lock:
                        self.text += char
                    print(char, end="", flush=True)
                self.last_time = time.time()
            time.sleep(0.01)

    def get_text_if_paused(self):
        with self.lock:
            if self.text and (time.time() - self.last_time > self.pause_time):
                t = self.text
                self.text = ""
                return t
        return None

class LiveAudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None

    def play_mp3_chunk(self, chunk):
        if chunk:
            audio = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
            if self.stream is None:
                self.stream = self.p.open(format=self.p.get_format_from_width(audio.sample_width),
                                          channels=audio.channels,
                                          rate=audio.frame_rate,
                                          output=True)
            self.stream.write(audio.raw_data)

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

async def listen(websocket, player):
    """Listen to the websocket for audio data and stream it."""
    while True:
        try:
            message = await websocket.recv()
            data = json.loads(message)
            if data.get("audio"):
                chunk = base64.b64decode(data["audio"])
                print(f"Received audio chunk of size: {len(chunk)}")
                player.play_mp3_chunk(chunk)
                with open("output_test.mp3", "ab") as f:
                    f.write(chunk)
            elif data.get('isFinal'):
                print("Audio stream finished.")
                break
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            break

async def text_to_speech_ws_streaming(voice_id, model_id):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"
    player = LiveAudioPlayer()
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "use_speaker_boost": False},
            "generation_config": {
                "chunk_length_schedule": [120, 160, 250, 290]
            },
            "xi_api_key": ELEVENLABS_API_KEY,
        }))
        input_handler = RealTimeInput(pause_time=1.0)
        input_handler.start()
        print("Start typing...")
        # Start listening for audio in the background
        listen_task = asyncio.create_task(listen(websocket, player))
        while input_handler.running:
            text = input_handler.get_text_if_paused()
            if text:
                print(f"\nSending: {text}")
                await websocket.send(json.dumps({"text": text, "flush": True}))
            await asyncio.sleep(0.1)
        await websocket.send(json.dumps({"text": ""}))  # End of text sequence
        await listen_task
    player.close()

if __name__ == "__main__":
    asyncio.run(text_to_speech_ws_streaming(voice_id, model_id))
    print("Audio saved to output_test.mp3")

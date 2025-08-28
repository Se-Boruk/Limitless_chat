import time
import numpy as np
import sounddevice as sd
import torch
from queue import Queue, Empty
from silero_vad import VADIterator, load_silero_vad
from transformers import AutoProcessor, MoonshineForConditionalGeneration

SAMPLING_RATE = 16000
CHUNK_SIZE = 512  # Frames per chunk from mic (Silero VAD requirement)

# Load Silero VAD ONNX model
vad_model = load_silero_vad(onnx=True)
vad_iterator = VADIterator(
    model=vad_model,
    sampling_rate=SAMPLING_RATE,
    threshold=0.5,
    min_silence_duration_ms=300,
)

print("Loading Moonshine model and processor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
dtype = torch.float32  # Force float32 even on CUDA

processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base")
model.to(device).to(dtype)
model.eval()

q = Queue()
speech_buffer = np.empty((0,), dtype=np.float32)
recording = False
last_print_time = time.time()
MIN_REFRESH_SECS = 0.2
MAX_SPEECH_SECS = 15.0

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio input status: {status}")
    q.put(indata[:, 0].copy())  # grab first channel, copy for thread safety

MIN_AUDIO_SAMPLES = int(3200)

def transcribe(audio_np):
    if len(audio_np) < MIN_AUDIO_SAMPLES:
        return ""  # skip too short input

    # Existing code follows...
    waveform = torch.from_numpy(audio_np).to(torch.float32)
    inputs = processor(waveform.numpy(), sampling_rate=SAMPLING_RATE, return_tensors="pt")
    inputs = {k: v.to(device).to(dtype) for k, v in inputs.items()}

    seq_lens = inputs['attention_mask'].sum(dim=-1)
    try:
        rate_factor = 6.5 / SAMPLING_RATE if SAMPLING_RATE > 0 else 1
        token_limit_float = (seq_lens * rate_factor).max().item()
        if not (np.isfinite(token_limit_float)) or token_limit_float <= 0:
            token_limit = 10
        else:
            token_limit = int(min(token_limit_float, 100))  # Cap to 100
    except Exception as e:
        print(f"Error computing token limit: {e}")
        token_limit = 10

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=token_limit)

    text = processor.decode(gen[0], skip_special_tokens=True)
    return text

print("Starting microphone stream... Press Ctrl+C to stop.")
with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, blocksize=CHUNK_SIZE, callback=audio_callback):
    try:
        while True:
            try:
                chunk = q.get(timeout=1)
            except Empty:
                continue

            speech_buffer = np.concatenate((speech_buffer, chunk))

            vad_result = vad_iterator(chunk)

            if vad_result:
                if "start" in vad_result and not recording:
                    recording = True
                    speech_buffer = np.copy(chunk)  # reset buffer at start
                    start_time = time.time()

                if "end" in vad_result and recording:
                    recording = False
                    # Transcribe full speech buffer on speech end
                    text = transcribe(speech_buffer)
                    print("\r" + " " * 80 + "\r", end="", flush=True)  # Clear line
                    print(f"Transcription: {text}", flush=True)
                    speech_buffer = np.empty((0,), dtype=np.float32)

            if recording:
                # Periodically update live transcription during speech
                now = time.time()
                if (now - last_print_time) > MIN_REFRESH_SECS:
                    text = transcribe(speech_buffer)
                    print("\r" + " " * 80 + "\r", end="", flush=True)  # Clear line
                    print(f"Transcription (live): {text}", end="", flush=True)
                    last_print_time = now

                # Safety cutoff for too long speech
                if (len(speech_buffer) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                    recording = False
                    text = transcribe(speech_buffer)
                    print("\r" + " " * 80 + "\r", end="", flush=True)
                    print(f"Transcription (timeout): {text}", flush=True)
                    speech_buffer = np.empty((0,), dtype=np.float32)

    except KeyboardInterrupt:
        print("\nExiting...")


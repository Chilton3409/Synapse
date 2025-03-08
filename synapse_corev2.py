#!/usr/bin/env python3
import vosk
import pyaudio
import re
from meta_ai_api import MetaAI
import os
import pyttsx3
import weakref
import gc
import cv2
import asyncio
import numpy as np
from scipy.signal import butter, lfilter

class VirtualAssistant:
    def __init__(self, api_token, speech_recog_model):
        self.api_token = api_token
        self.api = MetaAI(self.api_token)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        self.engine = pyttsx3.init()
        self.weak_engine = weakref.ref(self.engine)
        self.speech_recog_model = speech_recog_model
        self.recognizer = vosk.KaldiRecognizer(self.speech_recog_model, 16000)
        self.context = {}

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    async def process_audio(self):
        while True:
            chunk_size = int(16000 * 0.01)
            chunks = []
            for i in range(10):
                chunk = np.frombuffer(self.stream.read(chunk_size, exception_on_overflow=False), dtype=np.int16)
                chunks.append(chunk)
            concatenated_data = np.concatenate(chunks)

            # Apply bandpass filter to remove noise
            filtered_data = self.butter_bandpass_filter(concatenated_data, 100, 4000, 16000)

            # Normalize audio data
            normalized_data = filtered_data / np.max(np.abs(filtered_data))

            # Convert normalized data back to int16
            audio_data = (normalized_data * 32767).astype(np.int16)
            
            data = await self.process_audio_data(audio_data.tobytes())
            text = await self.speech_recognition(data)
            
            response = await self.create_response(text)
            
            self.text_to_speech(response)
            
           

    async def process_audio_data(self, data):
        noise_threshold = 10000
        attack = 2.00
        release = 2.00
        gain = int(1)
        amp = sum(abs(sample) for sample in data) / len(data)
        if amp < noise_threshold:
            pass
        return data

    async def speech_recognition(self, data):
        if self.recognizer.AcceptWaveform(data):
            result = self.recognizer.Result()
            if result != " ":
                final_text = result.split(":")[1].strip().replace('"', '').replace('}', '')
                if final_text.strip() != "" and len(final_text.strip()) > 2:
                    print(f"Human: {final_text}")
                    return final_text

    async def create_response(self, prompt):
        if prompt:
            response = self.api.prompt(message=prompt, new_conversation=False)
            message = response['message']
            print(f"Gen AI: {message}")
            return message

    def text_to_speech(self, prompt):
        if self.weak_engine():
            self.weak_engine().say(prompt)
            self.weak_engine().runAndWait()
        else:
            # force garbage collection
            pass
    async def start_stream(self):
        self.stream.start_stream()


    async def stop_stream(self):
        self.stream.stop_stream()

    async def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    async def run(self):
        await self.process_audio()

if __name__ == "__main__":
    api_token = os.environ.get("META_AI_TOKEN")
    # Load the Vosk model
    speech_recog_model = vosk.Model("vosk_models/vosk-model-small-en-us-0.15")
    assistant = VirtualAssistant(api_token, speech_recog_model)
    asyncio.run(assistant.run())
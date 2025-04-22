from typing import Dict
import os
import torch
import numpy as np
from TTS.utils.synthesizer import Synthesizer
import torchaudio
import tempfile

class EndpointHandler:
    def __init__(self):
        """
        Initialize the model and any other necessary components.
        This is run only once when the container starts.
        """
        # Get model paths from environment variables or use defaults
        model_path = os.environ.get("MODEL_PATH", "/app/model/checkpoint_1082083.pth")
        config_path = os.environ.get("CONFIG_PATH", "/app/model/config_tts.json")

        # Load the synthesizer model
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=torch.cuda.is_available()
        )

        # Set the target sample rate for output
        self.sample_rate = 16000

    def __call__(self, data: Dict) -> Dict:
        """
        This method is called every time the endpoint receives a request.
        Args:
            data (dict): A dictionary with a key "inputs" containing the input text.
        Returns:
            dict: A dictionary with a base64 encoded WAV audio string or binary payload.
        """
        # Parse input
        text = data.get("inputs", "")
        if not text:
            return {"error": "No input text provided."}

        # Run inference
        wav = self.synthesizer.tts(text)
        wav = np.array(wav, dtype=np.float32)
        wav = wav / max(abs(wav))  # Normalize
        wav_tensor = torch.tensor(wav).unsqueeze(0)

        # Resample to 16kHz
        resampled_wav = torchaudio.transforms.Resample(orig_freq=22050, new_freq=self.sample_rate)(wav_tensor)
        resampled_wav = resampled_wav.squeeze().numpy()

        # Convert to int16 PCM
        audio_int16 = (resampled_wav * 32767).astype(np.int16)

        # Save to temporary .wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            torchaudio.save(temp_wav_file.name, torch.from_numpy(audio_int16).unsqueeze(0), self.sample_rate)
            temp_wav_file.seek(0)
            audio_bytes = temp_wav_file.read()

        return {
            "content-type": "audio/wav",
            "body": audio_bytes,
        }

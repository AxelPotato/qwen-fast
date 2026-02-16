import torch
import soundfile as sf
import os
import uuid
import logging
# The qwen_tts package must be installed from the official repository or PyPI
from qwen_tts import Qwen3TTSModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTS_Engine")

class QwenTTSEngine:
    def __init__(self, model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        """
        Initializes the model on the RTX 3060 using BFloat16 and FlashAttention 2.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning("CUDA not found! Model will run extremely slowly on CPU.")
        
        logger.info(f"Loading model {model_path} onto {self.device}...")
        
        # Critical: Use bfloat16 for RTX 30-series (Ampere)
        # Use flash_attention_2 to minimize VRAM usage and maximize speed
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        logger.info("Model loaded successfully.")

    def clone_voice(self, text: str, ref_audio_path: str, output_dir: str, language: str = "auto") -> str:
        """
        Generates audio using the base model's voice cloning capability.
        """
        logger.info(f"Synthesizing text: '{text[:30]}...' using ref: {os.path.basename(ref_audio_path)}")
        
        # generate_voice_clone is the primary method for the Base model
        # It takes a list of texts, but we process one at a time for this API
        wavs, sr = self.model.generate_voice_clone(
            text=[text],
            language=[language],
            ref_audio=ref_audio_path,
        )
        
        # Generate unique filename for output
        filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Save audio using soundfile
        sf.write(output_path, wavs[0], sr)
        
        return filename
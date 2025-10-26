import torch
import scipy.io.wavfile
import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import os
import logging 
from pathlib import Path
import tempfile
from datetime import datetime
import warnings
import scipy
from scipy import signal


# setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    """Request model for TTS generation"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to convert to speech")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    emotion: Optional[str] = Field(default="neutral", description="Emotion context")
    is_emergency: bool = Field(default=False, description="Emergency speech (will effect tone and clarity)")
    preserve_prosody: bool = Field(default=True, description="Preserve natural speech prosody")


    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace")
        
        return v.strip()
    

class TTSResponse(BaseModel):
    """Respnse model for TTS generation"""
    original_text: str
    audio_file_path: str
    sample_rate: int
    duration_second: float
    model_used: str
    generation_time: float
    metadata: dict = {}


class NepaliTTSSystem:
    """Nepali Text to Speech conversion system using VITS model"""

    def __init__(self,
                 model_name: str = "tuskbyte/nepali_male_v1",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None
        ):
        """
        Initialize the Nepali TTS system.

        Args:
            model_name: Hugging face model identifier.
            cache_dir: Directory to cache downloaded models
            device: Device to run the model on ('cpu' or 'cuda'). If None, auto-detect.
        """
        self.model_name = model_name
        # self.cache_dir = cache_dir or os.path.expanduser("~/.cache/nepali_tts")
        self.cache_dir = cache_dir or "./audio_tts"
        os.makedirs(self.cache_dir, exist_ok=True)

        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'

        # creating cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing TTS system with model: {self.model_name} on device: {self.device}")

        self.model = None
        self.tokenizer = None
        self._load_model()


    def _load_model(self):
        """Load the VITS models and its tokenizer"""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            logger.info("Loading TTS model...")
            self.model = VitsModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32, # using float32 for compatibility
                cache_dir=self.cache_dir
            )

            self.model.to(self.device)


            # set to evaluation model for inference
            self.model.eval()
            logger.info("Model and tokenizer loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise RuntimeError(f"Failed to initialize TTS model: {e}")

        
    
    def _preprocess_text(self, text: str, is_emergency: bool = False) -> str:
        """
        Preprocess text for better TTS output
        Args:
            text: Input text
            is_emergency: Whether this is emergency speech
        Returns:
            Preprocessed text
        """

        #applying basic normalization
        text = text.strip()

        #for emergency speech, add emphasis markes 
        if is_emergency:
          # adding pauses for clarity in emergency situations
          text = text.replace("।", "। ")  # add space after nepali full stop
          text = text.replace("?", "? ")   # add space after question mark
          text = text.replace("!", "! ")   # add space after exclamation
          text = text.replace(",", ", ")   # add space after comma
            
            
            

        return text

    
    def generate_speech(self, request: TTSRequest) -> TTSResponse:
        """
        Generate speech from text using VITS model.
        Args:
            request: TTSRequest object with generation parameters
        Returns:
            TTSResponse with audio file path and metadata
        """
        start_time = datetime.now()

        try:
            logger.info(f"Generating speech for text: {request.text[:30]}...")
            # preprocess text
            preprocessed_text = self._preprocess_text(request.text, request.is_emergency)
            logger.info(f"Preprocessed text: {preprocessed_text}")

            # Tokenize input
            # inputs = self.tokenizer(preprocessed_text, return_tensors='pt').to(self.device)
            # updated
            inputs = self.tokenizer(
                preprocessed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move to device properly
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate speech
            with torch.no_grad():
                #setting seed for reproducible result
                if hasattr(self.model.config, "use_stochastic_duration_prediction"):
                    torch.manual_seed(42)

                output = self.model(**inputs,)
                waveform = output.waveform.squeeze().cpu().numpy() #numpy because easy to pass value to (librosa, soundfile, scipy)
                #for pytorch 
                # waveform = output.waveform.squeeze().cpu()
            
            if request.speed != 1.0:
                waveform  = self._modify_speed(waveform, request.speed)
            
            #saveing audio file
            output_path = self._save_audio(
                waveform, 
                self.model.config.sampling_rate,
                request.is_emergency
            )

            #calculationg duration
            duration = len(waveform) / self.model.config.sampling_rate
            generation_time = (datetime.now() - start_time).total_seconds()


            response = TTSResponse(
                original_text=request.text,
                audio_file_path=output_path,
                sample_rate=self.model.config.sampling_rate,
                duration_second=duration,
                model_used=self.model_name,
                generation_time=generation_time,
                metadata={
                    "processed_text": preprocessed_text,
                    "emotion": request.emotion,
                    "is_emergency": request.is_emergency,
                    "preserve_prosody": request.preserve_prosody,
                    "speed": request.speed
                }
            )

            logger.info(response)
            
            return response
          
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise RuntimeError(f"Failed to generate speech: {e}")


    def _modify_speed(self, waveform: np.ndarray, speed:float) -> np.ndarray:
        """
        Modify the speech speed using simple resampling.
        Args:
            waveform: Input audio waveform
            speed: Speed multipler (>1 = faster, <1 = slower)
        Returns:
            Speed-modified waveform
        """
        try:
            new_length = int(len(waveform) / speed)
            return signal.resample(waveform, new_length)
        
        except Exception as e:
            logger.warning(f"Failed to modify speed: {e}. Returning original waveform.")
            return waveform
    

    def _save_audio(self, waveform: np.ndarray, sample_rate: int, is_emergency:bool = False) -> str:
        """
        Save audio waveform to file
        Args:
            waveform: Audio waveform
            sample_rate: Sampling rate
            is_emergency: Whether this is emergency speech (affects filename)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "emergency_" if is_emergency else "speech_"
        filename = f"{prefix}{timestamp}.wav"

        output_dir = Path(self.cache_dir) / "generated_audio"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / filename

        # Ensure waveform is in correct format for spicy
        if waveform.dtype != np.int16:
            #converting to 19-bit PCM
            waveform_int16 = (waveform * 32767).astype(np.int16)
        else:
            waveform_int16 = waveform
        
        scipy.io.wavfile.write(str(output_path), sample_rate, waveform_int16)

        logger.info(f"Audio saved to: {output_path}")
        return str(output_path)


    def generate(self, text: str, **kwargs) -> str:
        """
        Simple interface for TTS generation
        Args:
            text: Text to convert to speech
            kwargs: Additional TTSRequest parameters (speed, emotion, etc.)
        Returns:
            Path to generated audio file
        """
        request = TTSRequest(text=text,speed=0.9, is_emergency=True, **kwargs)
        response = self.generate_speech(request)


        return response


    
    

        

    


    
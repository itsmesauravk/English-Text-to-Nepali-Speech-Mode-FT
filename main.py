# main.py
import torch
import logging
import streamlit as st
from scipy.io.wavfile import write
import numpy as np
from transformers import VitsModel, AutoProcessor

from translation import Translator, TranslationRequest
from speech import NepaliTTSSystem 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
logger.info(f"Using device: {device}")

# Load Fine-Tuned VITS
@st.cache_resource
def load_finetuned_model():
    model = VitsModel.from_pretrained("./fine_tuned_nepali_vits_v5.2")
    processor = AutoProcessor.from_pretrained("./fine_tuned_nepali_vits_v5.2")
    return model, processor

ft_model, ft_processor = load_finetuned_model()


# Load Pretrained Translator + TTS

translator = Translator()

@st.cache_resource
def load_pretrained_tts():
    return NepaliTTSSystem()

tts = load_pretrained_tts()

# Fine-tuned Nepali VITS inference
def generate_finetuned_tts(text: str, filename="ft_output.wav"):
    try:
        inputs = ft_processor(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = ft_model(**inputs).waveform
        output = output.cpu().squeeze().numpy()

        # Save
        write(filename, ft_model.config.sampling_rate, output)
        return filename
    except Exception as e:
        logger.error(f"Fine-tuned TTS error: {str(e)}")
        st.error(f"Fine-tuned TTS error: {str(e)}")
        return None

# Pretrained pipeline (Translate + TTS)
def run_pretrained_pipeline(english_text: str):
    try:
        # Translate
        translation_request = TranslationRequest(
            text=english_text,
            context="General conversation",
        )
        translated = translator.translate(translation_request).translated_text

        # TTS
        audio = tts.generate(translated)
        return translated, audio.audio_file_path
    except Exception as e:
        logger.error(f"Pretrained pipeline error: {str(e)}")
        st.error(f"Pretrained pipeline error: {str(e)}")
        return "", None


# Streamlit UI
def main():
    st.set_page_config(page_title="Nepali TTS Comparison", layout="wide")
    st.title("English → Nepali TTS (Pretrained vs Fine-Tuned)")

    english_text = st.text_area("Enter English text:", height=150)

    if st.button("Generate Both"):
        if english_text.strip() == "":
            st.warning("Please enter some text!")
        else:
            col1, col2 = st.columns(2)

            # Pretrained pipeline
            with col1:
                st.subheader("Pretrained Pipeline")
                with st.spinner("Running translation + TTS..."):
                    translated_text, pretrained_audio = run_pretrained_pipeline(english_text)
                if translated_text:
                    st.markdown(f"**Translated Nepali Text:** {translated_text}")
                if pretrained_audio:
                    st.audio(pretrained_audio)
                    with open(pretrained_audio, "rb") as f:
                        st.download_button("Download Pretrained Audio", f, "pretrained_output.wav")

            # Fine-tuned model
            with col2:
                st.subheader("Fine-Tuned Nepali VITS")
                with st.spinner("Generating audio with fine-tuned model..."):
                    if translated_text:
                        st.markdown(f"**Translated Nepali Text:** {translated_text}")
                    ft_audio = generate_finetuned_tts(translated_text)  
                if ft_audio:
                    st.audio(ft_audio)
                    with open(ft_audio, "rb") as f:
                        st.download_button("Download Fine-Tuned Audio", f, "finetuned_output.wav")


if __name__ == "__main__":
    main()



# call the ambulance
# how can i reach police station fast?
# can i use your phone for one call
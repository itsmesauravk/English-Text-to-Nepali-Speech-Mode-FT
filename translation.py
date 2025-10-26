from typing import Optional, List
from pydantic import BaseModel, Field, validator
from openai import OpenAI
import os
from enum import Enum
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class LanguageCode(str, Enum):
    """Supported language codes"""
    ENGLISH = "en"
    NEPALI = "ne"


class TranslationRequest(BaseModel):
    """Request model for translation"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to translate")
    source_language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Source language")
    target_language: LanguageCode = Field(default=LanguageCode.NEPALI, description="Target language")
    context: Optional[str] = Field(None, description="Additional context for better translation")
    formal_tone: bool = Field(default=True, description="Use formal tone in translation")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            logger.error("Text validation failed: Text cannot be empty or only whitespace")
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()
    


class TranslationResponse(BaseModel):
    """Response model for translation"""
    original_text: str
    translated_text: str
    source_language: LanguageCode
    target_language: LanguageCode
    confidence: Optional[float] = None
    alternative_translations: Optional[List[str]] = None



class PromptTemplate:
    """Template class for translation prompts"""
    
    @staticmethod
    def get_translation_prompt(
        text: str, 
        source_lang: str, 
        target_lang: str, 
        context: Optional[str] = None,
        formal_tone: bool = True
    ) -> str:
        """Generate translation prompt with context and tone specification"""
        
        tone_instruction = "formal and respectful" if formal_tone else "casual and natural"
        context_section = f"\n\nContext: {context}" if context else ""
        
        prompt = f"""You are an expert translator specializing in 'English' to 'Nepali' translation.

            Task: Translate the following 'English' text into 'Nepali'.

            Requirements:
            - Maintain the original meaning and intent
            - Use {tone_instruction} tone
            - Preserve any cultural nuances
            - Keep proper nouns in their original form unless they have established 'Nepali' equivalents
            - For emergency or urgent messages, prioritize clarity and immediacy{context_section}

            Text to translate: "{text}"

            Please provide only the 'Nepali' translation without explanations or additional commentary."""
                    
        return prompt


class Translator:
    """Translator with Pydantic support and better error handling"""
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Initialize the translator
        
        Args:
            model: OpenAI model to use for translation
        """
        self.client = OpenAI()
        self.model = model
    
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate text using the enhanced pipeline
        
        Args:
            request: TranslationRequest object with all translation parameters
            
        Returns:
            TranslationResponse object with translation results
        """
        try:
            # Generate prompt using template
            prompt = PromptTemplate.get_translation_prompt(
                text=request.text,
                source_lang=request.source_language.value,
                target_lang=request.target_language.value,
                context=request.context,
                formal_tone=request.formal_tone
            )

            # print("=== Generated Prompt ===")
            logger.info("=== Prompt Generated ===")
            
            # logger.info("Translation Prompt:\n" + prompt)

            logger.info("Starting translation API call...")
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent translations
                max_tokens=1000
            )
            logger.info("Translation API call completed.")
            
            translated_text = response.choices[0].message.content.strip()
            
            return TranslationResponse(
                original_text=request.text,
                translated_text=translated_text,
                source_language=request.source_language,
                target_language=request.target_language
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation failed: {str(e)}")
        


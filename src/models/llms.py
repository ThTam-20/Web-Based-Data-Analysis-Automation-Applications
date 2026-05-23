import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Nạp các API Key từ file .env
load_dotenv()

def load_llm(model_name):
    """Load Large Language Model (Hỗ trợ cả Free và Paid models)."""
    
    # OpenAI models (Paid)
    openai_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4"]
    if model_name in openai_models:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError(f"Bạn chưa cấu hình 'OPENAI_API_KEY' trong file .env để dùng {model_name}!")
            
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
            api_key=openai_api_key
        )
    
    # OpenRouter - Free OpenAI models
    elif model_name == "openai/gpt-oss-120b:free":
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("Bạn chưa cấu hình 'OPENROUTER_API_KEY' trong file .env!")
            
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    # Google Gemini models (Free/Paid)
    elif model_name in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.1-pro-preview"]:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Bạn chưa cấu hình 'GOOGLE_API_KEY' trong file .env!")
            
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            max_output_tokens=1000,
            api_key=google_api_key
        )
    
    else:
        supported_models = openai_models + [
            "openai/gpt-oss-120b:free",
            "gemini-2.5-flash",
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview"
        ]
        raise ValueError(
            f"Model '{model_name}' chưa được hỗ trợ.\n"
            f"Các model được hỗ trợ: {', '.join(supported_models)}"
        )
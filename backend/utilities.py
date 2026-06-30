from langchain_core.language_models import BaseChatModel
from backend.config import settings


def get_llm(provider=None, model=None, **kwargs) -> BaseChatModel:
    """
    Factory function that returns a configured LangChain BaseChatModel
    based on the LLM_PROVIDER env var. This is the ONLY place where
    LLM clients should be instantiated.

    Frontier providers: gemini, groq
    Local providers: ollama, llamacpp

    Extra kwargs (e.g. temperature, max_tokens) are passed through to
    the underlying chat model constructor.
    """
    if not provider:
        provider = settings.llm_provider.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        
        if not model:
            model="gemma-4-26b-a4b-it"

        return ChatGoogleGenerativeAI(
            model=model,    # gemma-4-31b-it Swith if needed
            api_key=settings.gemini_api_key,
            **kwargs,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        
        if not model:
            model="openai/gpt-oss-20b"

        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            **kwargs,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            **kwargs,
        )

    elif provider == "llamacpp":
        from langchain_community.chat_models import ChatLlamaCpp

        if not settings.llamacpp_model_path:
            raise ValueError("LLAMACPP_MODEL_PATH is not set in .env")

        return ChatLlamaCpp(
            model_path=settings.llamacpp_model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Must be one of: gemini, groq, ollama, llamacpp"
        )
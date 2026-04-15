import os
from dotenv import load_dotenv
load_dotenv()

def get_langfuse_handler(trace_name: str, user_id: str = "anonymous"):
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            session_id=user_id,
            tags=[trace_name],
        )
    except Exception:
        return None

def get_langfuse_client():
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        return None

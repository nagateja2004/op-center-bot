import os


os.environ.setdefault("CHECKPOINT_BACKEND", "sqlite")
os.environ["CHROMA_MODE"] = "local"
os.environ["CHAT_MEMORY_PATH"] = "/tmp/opcenter-chatbot-pytest.sqlite"

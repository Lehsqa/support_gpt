import os
from src.support_gpt import SupportGPT
from config.config import chat_history_path


if __name__ == "__main__":
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('OPENAI_API_KEY') == "":
        print("OPENAI_API_KEY is not set. Please add your key to config.py")
        exit(1)

    if not(os.path.exists(chat_history_path)):
        os.mkdir(chat_history_path)

    try:
        chat = SupportGPT()
        chat.run()
        chat.save_history()
    except Exception as e:
        print(e)

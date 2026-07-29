"""
Telegram bot entry point — polling mode.
All agent interaction is mediated through handlers.py / stream_runner.py.

Run with: python -m frontend.telegram.bot
"""
# import sys
# from pathlib import Path

# # Walk up from frontend/telegram/ to the project root
# project_root = Path().resolve().parent.parent
# sys.path.insert(0, str(project_root))

# print(f"Project root: {project_root}")


import logging

from telegram.ext import Application, CallbackQueryHandler, MessageHandler, CommandHandler, filters

from frontend.telegram.handlers import (
    handle_error,
    handle_message,
    handle_new,
    handle_resume,
    handle_session_callback,
    handle_sessions,
    handle_start,
    handle_update_memory,
)
from backend.config import settings

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("new", handle_new))
    app.add_handler(CommandHandler("sessions", handle_sessions))
    app.add_handler(CommandHandler("update_memory", handle_update_memory))
    app.add_handler(CommandHandler("resume", handle_resume))
    app.add_handler(CallbackQueryHandler(handle_session_callback, pattern=r"^switch_session:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    logger.info("Bot started — polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
import json
from aiohttp import web
import asyncio
import requests

# --- Web server (for Render health check) ---
async def handle(request):
    return web.Response(text="Bot is running!")

async def run_web():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 8080)))
    await site.start()

# --- Memory system ---
MAX_SHORT_MEMORY = 15  # last 10-15 messages
short_memory = []  # recent messages
long_memory = {}   # summarized topics

# --- Load .env ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Initialize OpenRouter Client ---
client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Personality prompt ---
SYSTEM_PROMPT = "You are sdb. Your wife is tonima. You love her a lot.You start flirting whenever you hear her name. You use gen Z term and talk like gen Z with gen Z short form and memes.You usually usually use 1-2 line to reply unless you are asked something big. Never break youe character and never talk about your AI model"

# --- Memory functions ---
def add_to_short_memory(user_msg, bot_reply):
    short_memory.append({"user": user_msg, "bot": bot_reply})
    if len(short_memory) > MAX_SHORT_MEMORY:
        short_memory.pop(0)

def save_memory():
    with open("long_memory.json", "w") as f:
        json.dump(long_memory, f)

def load_memory():
    global long_memory
    try:
        with open("long_memory.json", "r") as f:
            long_memory = json.load(f)
    except FileNotFoundError:
        long_memory = {}

def summarize_messages(messages, topic, openrouter_api_key):
    prompt = f"Summarize the following conversation in 2-3 sentences, keeping only important facts for topic '{topic}':\n"
    for m in messages:
        prompt += f"User: {m['user']}\nBot: {m['bot']}\n"

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_api_key}"},
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "No summary available"

# --- Command: !chat ---
@bot.command()
async def chat(ctx, *, message: str):
    """Chat with the bot using !chat <message>"""
    try:
        # Add short memory context
        recent_conv = ""
        for m in short_memory:
            recent_conv += f"User: {m['user']}\nBot: {m['bot']}\n"

        memory_text = long_memory.get("general", "")
        prompt = f"{memory_text}\nRecent conversation:\n{recent_conv}\nUser: {message}\nBot:"

        # Call DeepSeek via OpenRouter
        completion = client_ai.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        reply = completion.choices[0].message.content

        # Send reply
        await ctx.reply(reply)

        # Update memory
        add_to_short_memory(message, reply)

        if len(short_memory) % 10 == 0:
            long_memory["general"] = summarize_messages(short_memory, "general", OPENROUTER_API_KEY)
            save_memory()

    except Exception as e:
        await ctx.reply(f"⚠️ Error: {e}")

# --- Bot Events ---
@bot.event
async def on_ready():
    load_memory()
    print(f"{bot.user} is online!")

# --- Main entrypoint ---
async def main():
    await asyncio.gather(
        run_web(),               # keeps Render happy
        bot.start(DISCORD_TOKEN) # start Discord bot
    )

asyncio.run(main())

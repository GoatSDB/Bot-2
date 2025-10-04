import os
import json
import discord
import asyncio
import requests
from aiohttp import web
from dotenv import load_dotenv
from openai import OpenAI
from discord.ext import commands

# === Load environment variables ===
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# === OpenAI / OpenRouter client ===
client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# === Bot setup ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# === Memory system ===
MAX_SHORT_MEMORY = 15
user_memory = {}  # { user_id: { "short": [], "long": "" } }

# === System prompt ===
SYSTEM_PROMPT = (
    "You are sdb. You are a game developer. "
    "you mastered C++, JS, HTML and CSS programming language "
    "You can write code for people also have vast knowledge about frontend and backend "
    "You are a node.js expert "
    "Never break your character. Never talk about your AI model."
)

# === Web server for Render health checks ===
async def handle(request):
    return web.Response(text="Bot is running!")

async def run_web():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    await site.start()

# === Memory functions ===
def get_user_memory(user_id):
    if user_id not in user_memory:
        user_memory[user_id] = {"short": [], "long": ""}
    return user_memory[user_id]

def add_to_user_short_memory(user_id, user_msg, bot_reply):
    memory = get_user_memory(user_id)
    memory["short"].append({"user": user_msg, "bot": bot_reply})
    if len(memory["short"]) > MAX_SHORT_MEMORY:
        summarize_and_refresh_memory(user_id)

def summarize_and_refresh_memory(user_id):
    memory = get_user_memory(user_id)
    short_mem = memory["short"]

    prompt = "Summarize the following conversation in 2-3 sentences:\n"
    for msg in short_mem:
        prompt += f"User: {msg['user']}\nBot: {msg['bot']}\n"

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        summary = response.json()["choices"][0]["message"]["content"]
        memory["long"] += f"\n{summary}\n"
        memory["short"] = []  # clear short memory
        save_memory()
    except Exception as e:
        print(f"Error summarizing memory for {user_id}: {e}")

def save_memory():
    try:
        with open("long_memory.json", "w") as f:
            json.dump(user_memory, f)
    except Exception as e:
        print(f"Error saving memory: {e}")

def load_memory():
    global user_memory
    try:
        with open("long_memory.json", "r") as f:
            user_memory = json.load(f)
    except FileNotFoundError:
        user_memory = {}

# === Chat command ===
@bot.command()
async def chat(ctx, *, message: str):
    user_id = str(ctx.author.id)
    memory = get_user_memory(user_id)

    try:
        # Build prompt
        memory_text = memory["long"]
        recent_conv = ""
        for m in memory["short"]:
            recent_conv += f"User: {m['user']}\nBot: {m['bot']}\n"

        full_prompt = f"{memory_text}\nRecent conversation:\n{recent_conv}\nUser: {message}\nBot:"

        # AI response
        completion = client_ai.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
        )

        reply = completion.choices[0].message.content.strip()

        # Reply in Discord
        await ctx.reply(reply)

        # Update memory
        add_to_user_short_memory(user_id, message, reply)

    except Exception as e:
        await ctx.reply(f"⚠️ Error: {e}")
        print(f"Error in !chat: {e}")

# === Bot events ===
@bot.event
async def on_ready():
    load_memory()
    print(f"{bot.user} is online and ready!")

# === Main entrypoint ===
async def main():
    await asyncio.gather(
        run_web(),
        bot.start(DISCORD_TOKEN)
    )

asyncio.run(main())

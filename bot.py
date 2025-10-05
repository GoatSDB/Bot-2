import os
import json
import discord
import asyncio
import requests
from io import BytesIO
from aiohttp import web
from dotenv import load_dotenv
from openai import OpenAI
from discord.ext import commands

# === Load environment variables ===
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
A4F_API_KEY = os.getenv("A4F_API_KEY")  # <== new key for imagen-4

# === OpenRouter client ===
client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# === Discord setup ===
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# === Web health server ===
async def handle(request):
    return web.Response(text="Bot is running!")

async def run_web():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    await site.start()

# === Personality definitions ===
PERSONALITIES = {
    "chat": {
        "system_prompt": (
            "You are SDB, a professional game developer and coding expert. "
            "You master C++, JS, HTML, CSS, Node.js, backend/frontend. "
            "You never break character."
        ),
        "memory_file": "memory_sdb.json",
        "max_short_memory": 15,
    },
    "mint": {
        "system_prompt": (
            "You are Mint, a chill and friendly assistant. "
            "You help users relax, stay positive, and have a kind tone."
        ),
        "memory_file": "memory_mint.json",
        "max_short_memory": 15,
    },
    "art": {
        "system_prompt": (
            "You are Artie, an imaginative AI artist. "
            "You create detailed, beautiful images from user prompts using Imagen-4."
        ),
        "memory_file": "memory_art.json",
        "max_short_memory": 10,
    },
}

# === Global memory ===
user_memory = {name: {} for name in PERSONALITIES}

def get_user_memory(persona, user_id):
    if user_id not in user_memory[persona]:
        user_memory[persona][user_id] = {"short": [], "long": ""}
    return user_memory[persona][user_id]

def save_memory(persona):
    try:
        with open(PERSONALITIES[persona]["memory_file"], "w") as f:
            json.dump(user_memory[persona], f)
    except Exception as e:
        print(f"[ERROR] Saving memory for {persona}: {e}")

def load_memory(persona):
    try:
        with open(PERSONALITIES[persona]["memory_file"], "r") as f:
            user_memory[persona] = json.load(f)
    except FileNotFoundError:
        user_memory[persona] = {}

def safe_api_call(model, messages):
    """For text models."""
    try:
        completion = client_ai.chat.completions.create(model=model, messages=messages)
        if not hasattr(completion, "choices") or not completion.choices:
            return None
        content = getattr(completion.choices[0].message, "content", None)
        if not content or content.strip().startswith("{"):
            return None
        return content.strip()
    except Exception as e:
        print(f"[API ERROR] {e}")
        return None

def summarize_and_refresh_memory(persona, user_id):
    memory = get_user_memory(persona, user_id)
    short_mem = memory["short"]
    if not short_mem:
        return
    prompt = "Summarize the following conversation in 2-3 sentences:\n"
    for msg in short_mem:
        prompt += f"User: {msg['user']}\nBot: {msg['bot']}\n"
    summary = safe_api_call(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    if summary:
        memory["long"] += f"\n{summary}\n"
        memory["short"] = []
        save_memory(persona)
    else:
        print(f"[WARN] Failed to summarize memory for {user_id} ({persona})")

def add_to_user_short_memory(persona, user_id, user_msg, bot_reply):
    memory = get_user_memory(persona, user_id)
    memory["short"].append({"user": user_msg, "bot": bot_reply})
    if len(memory["short"]) > PERSONALITIES[persona]["max_short_memory"]:
        summarize_and_refresh_memory(persona, user_id)

# === Shared chat handler ===
async def handle_chat(ctx, persona, message):
    if persona == "art":
        # Handle image generation instead of chat
        await handle_image(ctx, message)
        return

    user_id = str(ctx.author.id)
    memory = get_user_memory(persona, user_id)
    persona_cfg = PERSONALITIES[persona]

    memory_text = memory["long"]
    recent_conv = "\n".join(
        [f"User: {m['user']}\nBot: {m['bot']}" for m in memory["short"]]
    )
    full_prompt = f"{memory_text}\nRecent conversation:\n{recent_conv}\nUser: {message}\nBot:"

    reply = safe_api_call(
        model="deepseek/deepseek-chat-v3.1:free",
        messages=[
            {"role": "system", "content": persona_cfg["system_prompt"]},
            {"role": "user", "content": full_prompt},
        ],
    )

    if not reply:
        await ctx.reply("⚠️ Sorry, I couldn’t process that request. Please try again later.")
        return

    await ctx.reply(reply)
    add_to_user_short_memory(persona, user_id, message, reply)

# === Image generation ===
async def handle_image(ctx, prompt: str):
    await ctx.typing()
    try:
        response = requests.post(
            "https://api.a4f.ai/v1/images/generations",
            headers={"Authorization": f"Bearer {A4F_API_KEY}"},
            json={
                "model": "provider-4/imagen-4",
                "prompt": prompt,
                "size": "1024x1024",
            },
        )
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            img_url = data["data"][0].get("url")
            if img_url:
                await ctx.reply(f"🎨 **Generated Image:** {prompt}\n{img_url}")
                return
        await ctx.reply("⚠️ Failed to generate image. Please try again later.")
    except Exception as e:
        await ctx.reply(f"⚠️ Error: {e}")
        print(f"[IMAGE ERROR] {e}")

# === Commands ===
@bot.command()
async def chat(ctx, *, message: str):
    await handle_chat(ctx, "chat", message)

@bot.command()
async def mint(ctx, *, message: str):
    await handle_chat(ctx, "mint", message)

@bot.command()
async def art(ctx, *, message: str):
    await handle_chat(ctx, "art", message)

# === Bot events ===
@bot.event
async def on_ready():
    for persona in PERSONALITIES:
        load_memory(persona)
    print(f"{bot.user} is online with {len(PERSONALITIES)} personalities ready!")

# === Main entry ===
async def main():
    await asyncio.gather(run_web(), bot.start(DISCORD_TOKEN))

asyncio.run(main())

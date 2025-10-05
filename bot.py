# bot.py
import os
import json
import asyncio
import base64
import io
from dotenv import load_dotenv
import aiohttp
import requests
import discord
from aiohttp import web
from openai import OpenAI
from discord.ext import commands

load_dotenv()

# === Config / env ===
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
A4F_API_KEY = os.getenv("A4F_API_KEY")
MOD_ROLE_NAME = os.getenv("MOD_ROLE_NAME", "Admin")  # role that can moderate (or admins)

# A4F base URL is required to be hard-coded per your request:
A4F_API_BASE = "https://api.a4f.co/v1"

# === OpenRouter / OpenAI client used for text chat ===
client_ai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# === Discord bot setup ===
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# === Memory (per-personality, per-user) ===
MEMORY_FILE = "long_memory.json"
MAX_SHORT_MEMORY = 15

user_memory = {
    "chat": {},
    "mint": {},
    "art": {},
}

def load_memory():
    global user_memory
    try:
        with open(MEMORY_FILE, "r") as f:
            user_memory = json.load(f)
            # ensure keys exist
            for p in ("chat", "mint", "art"):
                user_memory.setdefault(p, {})
    except FileNotFoundError:
        user_memory = {"chat": {}, "mint": {}, "art": {}}

def save_memory():
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(user_memory, f)
    except Exception as e:
        print("Error saving memory:", e)

def get_user_memory(persona: str, user_id: str):
    mem = user_memory.setdefault(persona, {})
    if user_id not in mem:
        mem[user_id] = {"short": [], "long": ""}
    return mem[user_id]

def add_to_short_memory(persona: str, user_id: str, user_msg: str, bot_reply: str):
    m = get_user_memory(persona, user_id)
    m["short"].append({"user": user_msg, "bot": bot_reply})
    if len(m["short"]) > MAX_SHORT_MEMORY:
        summarize_and_refresh(persona, user_id)
    save_memory()

def summarize_and_refresh(persona: str, user_id: str):
    m = get_user_memory(persona, user_id)
    if not m["short"]:
        return
    prompt = "Summarize the following conversation in 2-3 sentences:\n"
    for pair in m["short"]:
        prompt += f"User: {pair['user']}\nBot: {pair['bot']}\n"
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        data = resp.json()
        summary = None
        # try to extract content robustly
        if isinstance(data, dict):
            choices = data.get("choices")
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                summary = msg.get("content")
        if summary:
            m["long"] += f"\n{summary}\n"
            m["short"] = []
            save_memory()
        else:
            print("Summarize: unexpected response", data)
    except Exception as e:
        print("Error summarizing memory:", e)

# === Personality system prompts ===
PERSONALITY_PROMPTS = {
    "chat": (
        "You are sdb. You are a game developer. you mastered C++, JS, HTML and CSS programming language. "
        "You can write code for people and have vast knowledge about frontend and backend. "
        "You are a Node.js expert. Never break your character. Never talk about your AI model."
    ),
    "mint": (
        "You are nkt. A funny helpful girl. "
        "you mostly talk in modern style, kinda like Gen Z. "
        "you are humorous and makes joke a lot. Never mention being an AI."
    ),
    "art": (
        "You are Artie, an imaginative AI artist. "
        "When asked to create images, return concise creative prompts and generate images using the Imagen-4 API."
    ),
}

# === Safe wrapper for text model calls ===
def safe_chat_completion(model: str, messages: list, timeout: int = 30):
    """Call the OpenRouter chat completion and return safe string or None."""
    try:
        completion = client_ai.chat.completions.create(model=model, messages=messages)
        # Try a few extraction paths
        content = None
        try:
            # object-like
            content = completion.choices[0].message.content
        except Exception:
            # dict-like fallback
            try:
                if isinstance(completion, dict):
                    choices = completion.get("choices")
                    if choices:
                        content = choices[0].get("message", {}).get("content")
            except Exception:
                content = None
        if not content:
            print("[WARN] No content found in completion:", completion)
            return None
        text = content.strip()
        # Guard against raw JSON error bodies
        if text.startswith("{") and ("status" in text or "error" in text):
            print("[WARN] Chat returned JSON-like error:", text)
            return None
        return text
    except Exception as e:
        print("safe_chat_completion error:", e)
        return None

async def generate_text_reply(persona: str, user_id: str, user_message: str):
    memory = get_user_memory(persona, user_id)
    memory_text = memory.get("long", "")
    recent = ""
    for item in memory.get("short", []):
        recent += f"User: {item['user']}\nBot: {item['bot']}\n"
    full_prompt = f"{memory_text}\nRecent conversation:\n{recent}\nUser: {user_message}\nBot:"
    messages = [
        {"role": "system", "content": PERSONALITY_PROMPTS[persona]},
        {"role": "user", "content": full_prompt},
    ]
    reply = safe_chat_completion(model="deepseek/deepseek-chat-v3.1:free", messages=messages)
    if reply:
        add_to_short_memory(persona, user_id, user_message, reply)
        return reply
    return "⚠️ Sorry — I couldn't get a response from the AI right now. Try again shortly."

# === Image generation via A4F (async) ===
async def generate_image_a4f(prompt: str):
    """
    Uses A4F Imagen-4 endpoint at https://api.a4f.co/v1/images/generations.
    Returns tuple (bytes_data, filename) on success, or (None, error_message) on failure.
    """
    url = f"{A4F_API_BASE}/images/generations"
    headers = {
        "Authorization": f"Bearer {A4F_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "provider-4/imagen-4",
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return None, f"API error {resp.status}: {text[:1000]}"
                data = await resp.json()
    except Exception as e:
        return None, f"HTTP error when calling A4F: {e}"

    # Expected: data["data"][0]["url"] or ["b64_json"] depending on API
    entry = None
    if isinstance(data, dict):
        arr = data.get("data")
        if arr and len(arr) > 0:
            entry = arr[0]

    if not entry:
        return None, f"Unexpected A4F response: {data}"

    # If the API returns a base64 image:
    b64 = entry.get("b64_json") or entry.get("b64") or entry.get("base64")
    if b64:
        try:
            img_bytes = base64.b64decode(b64)
            filename = "image.png"
            return img_bytes, filename
        except Exception as e:
            return None, f"Could not decode base64 image: {e}"

    # If the API returns a URL:
    url_out = entry.get("url") or entry.get("image") or entry.get("src")
    if url_out:
        # fetch bytes
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url_out, timeout=60) as r:
                    if r.status != 200:
                        return None, f"Failed to fetch image from url ({r.status})."
                    img_bytes = await r.read()
                    # determine extension from content-type (fallback to png)
                    content_type = r.headers.get("Content-Type", "")
                    ext = "png"
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = "jpg"
                    filename = f"image.{ext}"
                    return img_bytes, filename
        except Exception as e:
            return None, f"Error downloading generated image: {e}"

    return None, "No image data found in response."

# === Helper: parse durations like 10m, 1h, 30s, 2d ===
def parse_duration_to_seconds(s: str) -> int:
    if not s:
        return 0
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    unit = s[-1]
    try:
        val = int(s[:-1])
    except Exception:
        return 0
    if unit == "s":
        return val
    if unit == "m":
        return val * 60
    if unit == "h":
        return val * 3600
    if unit == "d":
        return val * 86400
    return 0

# === Moderation helpers ===
def user_has_mod_role(member: discord.Member):
    if member.guild_permissions.administrator:
        return True
    return any(r.name == MOD_ROLE_NAME for r in member.roles)

async def ensure_muted_role(guild: discord.Guild) -> discord.Role:
    role = discord.utils.get(guild.roles, name="Muted")
    if role:
        return role
    try:
        role = await guild.create_role(name="Muted", reason="Create role for muting users")
    except Exception as e:
        print("Could not create Muted role:", e)
        return None
    # Deny send_messages in text channels, speak in voice channels
    for ch in guild.channels:
        try:
            if isinstance(ch, discord.TextChannel):
                await ch.set_permissions(role, send_messages=False, add_reactions=False)
            elif isinstance(ch, discord.VoiceChannel):
                await ch.set_permissions(role, speak=False, connect=False)
        except Exception as e:
            print(f"Could not set perms in {ch.name}: {e}")
    return role

# === Commands: chat, mint, art ===
@bot.command()
async def chat(ctx, *, message: str):
    """SDB developer persona (text)."""
    reply = await generate_text_reply("chat", str(ctx.author.id), message)
    await ctx.reply(reply)

@bot.command()
async def mint(ctx, *, message: str):
    """Gen Z / funny persona (text)."""
    reply = await generate_text_reply("mint", str(ctx.author.id), message)
    await ctx.reply(reply)

@bot.command()
async def art(ctx, *, message: str):
    """Art persona: generates images when asked, otherwise behaves like chat persona."""
    # Treat messages that ask to create/generate/draw as image prompts
    lower = message.lower()
    is_image = any(word in lower for word in ("create", "generate", "draw", "image", "picture", "make a"))
    if is_image:
        await ctx.trigger_typing()
        img_bytes, info = await generate_image_a4f(message)
        if img_bytes:
            # send as file
            file = discord.File(io.BytesIO(img_bytes), filename=info)
            try:
                await ctx.reply(file=file)
            except Exception:
                # fallback to upload via channel.send
                await ctx.send(file=file)
        else:
            # info contains error message
            await ctx.reply(f"⚠️ Image generation failed: {info}")
    else:
        reply = await generate_text_reply("art", str(ctx.author.id), message)
        await ctx.reply(reply)

# === Moderation commands ===
@bot.command()
async def ban(ctx, member: discord.Member, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        return await ctx.reply("⛔ You don't have permission to use this.")
    try:
        await member.ban(reason=reason)
        await ctx.reply(f"✅ Banned {member} ({member.id}). Reason: {reason}")
    except Exception as e:
        await ctx.reply("⚠️ Failed to ban the member (check role hierarchy & permissions).")
        print("ban error:", e)

@bot.command()
async def kick(ctx, member: discord.Member, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        return await ctx.reply("⛔ You don't have permission to use this.")
    try:
        await member.kick(reason=reason)
        await ctx.reply(f"✅ Kicked {member} ({member.id}). Reason: {reason}")
    except Exception as e:
        await ctx.reply("⚠️ Failed to kick the member.")
        print("kick error:", e)

@bot.command()
async def mute(ctx, member: discord.Member, duration: str = None, *, reason: str = None):
    """
    Usage:
      !mute @user 10m reason text
      !mute @user 30s
      !mute @user    (indefinite)
    """
    if not user_has_mod_role(ctx.author):
        return await ctx.reply("⛔ You don't have permission to use this.")
    role = await ensure_muted_role(ctx.guild)
    if not role:
        return await ctx.reply("⚠️ Could not create/find Muted role (check bot perms).")
    try:
        await member.add_roles(role, reason=reason)
        msg = f"🔇 Muted {member.mention}"
        seconds = 0
        if duration:
            seconds = parse_duration_to_seconds(duration)
            if seconds > 0:
                msg += f" for {duration}"
        if reason:
            msg += f" | Reason: {reason}"
        await ctx.reply(msg)
        # schedule unmute if timed
        if seconds and seconds > 0:
            async def _delayed_unmute():
                await asyncio.sleep(seconds)
                try:
                    if role in member.roles:
                        await member.remove_roles(role, reason="Auto unmute after timer")
                        try:
                            await ctx.send(f"🔊 {member.mention} has been automatically unmuted.")
                        except Exception:
                            pass
                except Exception as e:
                    print("Auto-unmute failed:", e)
            bot.loop.create_task(_delayed_unmute())
    except Exception as e:
        await ctx.reply("⚠️ Failed to mute the member.")
        print("mute error:", e)

@bot.command()
async def unmute(ctx, member: discord.Member, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        return await ctx.reply("⛔ You don't have permission to use this.")
    role = discord.utils.get(ctx.guild.roles, name="Muted")
    if not role:
        return await ctx.reply("⚠️ Muted role does not exist.")
    try:
        if role in member.roles:
            await member.remove_roles(role, reason=reason)
            await ctx.reply(f"🔊 Unmuted {member}.")
        else:
            await ctx.reply("⚠️ Member is not muted.")
    except Exception as e:
        await ctx.reply("⚠️ Failed to unmute the member.")
        print("unmute error:", e)

@bot.command()
async def purge(ctx, amount: int):
    if not user_has_mod_role(ctx.author):
        return await ctx.reply("⛔ You don't have permission to use this.")
    if amount < 1 or amount > 2000:
        return await ctx.reply("⚠️ Amount must be between 1 and 2000.")
    try:
        deleted = await ctx.channel.purge(limit=amount + 1)
        await ctx.send(f"🧹 Deleted {len(deleted)-1} messages.", delete_after=5)
    except Exception as e:
        await ctx.reply("⚠️ Failed to purge messages (check Manage Messages permission).")
        print("purge error:", e)

# === Health webserver ===
async def health(request):
    return web.Response(text="Bot is running!")

async def run_web():
    app = web.Application()
    app.router.add_get("/", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    await site.start()

# === Events ===
@bot.event
async def on_ready():
    load_memory()
    print(f"{bot.user} is online — personalities: chat, mint, art")

# === Entrypoint ===
async def main():
    await asyncio.gather(run_web(), bot.start(DISCORD_TOKEN))

if __name__ == "__main__":
    asyncio.run(main())

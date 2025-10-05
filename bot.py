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
MOD_ROLE_NAME = os.getenv("MOD_ROLE_NAME", "Admin")

# Hard-coded A4F API base URL
A4F_API_BASE = "https://api.a4f.co/v1"

client_ai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# === Discord setup ===
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# === Memory ===
MEMORY_FILE = "long_memory.json"
MAX_SHORT_MEMORY = 15
user_memory = {"chat": {}, "mint": {}, "art": {}}

def load_memory():
    global user_memory
    try:
        with open(MEMORY_FILE, "r") as f:
            user_memory = json.load(f)
            for p in ("chat", "mint", "art"):
                user_memory.setdefault(p, {})
    except FileNotFoundError:
        user_memory = {"chat": {}, "mint": {}, "art": {}}

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(user_memory, f)

def get_user_memory(persona, user_id):
    mem = user_memory.setdefault(persona, {})
    if user_id not in mem:
        mem[user_id] = {"short": [], "long": ""}
    return mem[user_id]

def add_to_short_memory(persona, user_id, user_msg, bot_reply):
    m = get_user_memory(persona, user_id)
    m["short"].append({"user": user_msg, "bot": bot_reply})
    if len(m["short"]) > MAX_SHORT_MEMORY:
        summarize_and_refresh(persona, user_id)
    save_memory()

def summarize_and_refresh(persona, user_id):
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
        choices = data.get("choices")
        if choices:
            msg = choices[0].get("message", {})
            summary = msg.get("content")
            if summary:
                m["long"] += f"\n{summary}\n"
                m["short"] = []
                save_memory()
    except Exception as e:
        print("Summarize failed:", e)

# === Personalities ===
PERSONALITY_PROMPTS = {
    "chat": (
        "You are sdb. You are a game developer, expert in C++, JS, HTML, CSS, and Node.js. "
        "You write efficient code and never break character or mention AI models."
    ),
    "mint": (
        "You are nkt, a funny helpful Gen Z girl. You talk in a modern, casual, humorous way. "
        "You joke around but are still helpful and smart. Never mention being an AI."
    ),
    "art": (
        "You are Artie, an imaginative artist AI who generates stunning visuals. "
        "When prompted to create images, respond creatively and use Imagen-4 via A4F API."
    ),
}

def safe_chat_completion(model, messages, timeout=30):
    try:
        completion = client_ai.chat.completions.create(model=model, messages=messages)
        content = completion.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        print("safe_chat_completion error:", e)
        return None

async def generate_text_reply(persona, user_id, user_message):
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
    reply = safe_chat_completion("deepseek/deepseek-chat-v3.1:free", messages)
    if reply:
        add_to_short_memory(persona, user_id, user_message, reply)
        return reply
    return "⚠️ Sorry — I couldn’t get a response right now."

# === ✅ FIXED IMAGE GENERATION ===
async def generate_image_a4f(prompt: str):
    """Call A4F Imagen-4 API and return (bytes, filename) or (None, error)."""
    url = f"{A4F_API_BASE}/images/generate"  # Correct A4F-style endpoint
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
                    return None, f"⚠️ A4F API error {resp.status}: {text[:500]}"
                data = json.loads(text)
    except Exception as e:
        return None, f"⚠️ HTTP error contacting A4F: {e}"

    # Handle possible A4F response formats
    try:
        if isinstance(data, dict):
            # Check for base64
            b64 = (
                data.get("b64_json")
                or data.get("b64")
                or data.get("image_base64")
            )
            if b64:
                img_bytes = base64.b64decode(b64)
                return img_bytes, "generated.png"

            # Check for URL
            url_out = (
                data.get("url")
                or data.get("image_url")
                or data.get("image")
                or data.get("data", [{}])[0].get("url")
            )
            if url_out:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url_out, timeout=30) as img_resp:
                        if img_resp.status != 200:
                            return None, f"⚠️ Failed to fetch image ({img_resp.status})."
                        img_bytes = await img_resp.read()
                        return img_bytes, "generated.png"

        return None, f"⚠️ Unexpected A4F response: {str(data)[:500]}"
    except Exception as e:
        return None, f"⚠️ Parsing error: {e}"

# === Commands ===
@bot.command()
async def chat(ctx, *, message: str):
    reply = await generate_text_reply("chat", str(ctx.author.id), message)
    await ctx.reply(reply)

@bot.command()
async def mint(ctx, *, message: str):
    reply = await generate_text_reply("mint", str(ctx.author.id), message)
    await ctx.reply(reply)

@bot.command()
async def art(ctx, *, message: str):
    lower = message.lower()
    is_image = any(w in lower for w in ["create", "generate", "draw", "image", "picture", "paint", "make a"])
    if is_image:
        await ctx.trigger_typing()
        img_bytes, info = await generate_image_a4f(message)
        if img_bytes:
            file = discord.File(io.BytesIO(img_bytes), filename=info)
            await ctx.reply(file=file)
        else:
            await ctx.reply(info)
    else:
        reply = await generate_text_reply("art", str(ctx.author.id), message)
        await ctx.reply(reply)

# === Moderation ===
def user_has_mod_role(member):
    if member.guild_permissions.administrator:
        return True
    return any(r.name == MOD_ROLE_NAME for r in member.roles)

async def ensure_muted_role(guild):
    role = discord.utils.get(guild.roles, name="Muted")
    if role:
        return role
    role = await guild.create_role(name="Muted", reason="Create role for muting users")
    for ch in guild.channels:
        try:
            await ch.set_permissions(role, send_messages=False, speak=False)
        except:
            pass
    return role

def parse_duration_to_seconds(s: str) -> int:
    if not s:
        return 0
    s = s.strip().lower()
    unit = s[-1]
    try:
        val = int(s[:-1])
    except:
        return 0
    if unit == "s": return val
    if unit == "m": return val * 60
    if unit == "h": return val * 3600
    if unit == "d": return val * 86400
    return 0

@bot.command()
async def mute(ctx, member: discord.Member, duration: str = None, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        return await ctx.reply("⛔ You don't have permission to use this.")
    role = await ensure_muted_role(ctx.guild)
    await member.add_roles(role, reason=reason)
    msg = f"🔇 Muted {member.mention}"
    seconds = parse_duration_to_seconds(duration) if duration else 0
    if seconds > 0:
        msg += f" for {duration}"
    if reason:
        msg += f" | Reason: {reason}"
    await ctx.reply(msg)

    if seconds > 0:
        async def unmute_later():
            await asyncio.sleep(seconds)
            if role in member.roles:
                await member.remove_roles(role, reason="Auto unmute after timer")
                await ctx.send(f"🔊 {member.mention} has been automatically unmuted.")
        bot.loop.create_task(unmute_later())

@bot.command()
async def purge(ctx, amount: int):
    if not user_has_mod_role(ctx.author):
        return await ctx.reply("⛔ You don't have permission.")
    deleted = await ctx.channel.purge(limit=amount + 1)
    await ctx.send(f"🧹 Deleted {len(deleted)-1} messages.", delete_after=5)

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

@bot.event
async def on_ready():
    load_memory()
    print(f"{bot.user} is online — personalities: chat, mint, art")

async def main():
    await asyncio.gather(run_web(), bot.start(DISCORD_TOKEN))

if __name__ == "__main__":
    asyncio.run(main())

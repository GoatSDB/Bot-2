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
MOD_ROLE_NAME = os.getenv("MOD_ROLE_NAME", "Moderator")

# === OpenRouter Client ===
client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# === Discord Setup ===
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# === Web server for health checks ===
async def handle(request):
    return web.Response(text="Bot is running!")

async def run_web():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    await site.start()

# === Personalities ===
PERSONALITIES = {
    "chat": {
        "system_prompt": (
            "You are SDB, a skilled game developer and coding expert. "
            "You master C++, JS, HTML, CSS, Node.js and backend/frontend development. "
            "You are confident, efficient, and never break character."
        ),
        "memory_file": "memory_sdb.json",
        "max_short_memory": 15
    },
    "mint": {
        "system_prompt": (
            "You are nkt. A funny helpful girl."
            "you mostly talk in modern style,  kinda like gen Z"
            "you are humorous and makes joke a lot"
        ), 
        "memory_file": "memory_mint.json",
        "max_short_memory": 15
    }
}

# === Global Memory Containers ===
user_memory = {name: {} for name in PERSONALITIES}

# === Memory Helpers ===
def get_user_memory(persona: str, user_id: str):
    if user_id not in user_memory[persona]:
        user_memory[persona][user_id] = {"short": [], "long": ""}
    return user_memory[persona][user_id]

def save_memory(persona: str):
    try:
        with open(PERSONALITIES[persona]["memory_file"], "w") as f:
            json.dump(user_memory[persona], f)
    except Exception as e:
        print(f"[ERROR] Saving memory for {persona}: {e}")

def load_memory(persona: str):
    try:
        with open(PERSONALITIES[persona]["memory_file"], "r") as f:
            user_memory[persona] = json.load(f)
    except FileNotFoundError:
        user_memory[persona] = {}

def safe_api_call(model: str, messages: list):
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

def summarize_and_refresh_memory(persona: str, user_id: str):
    memory = get_user_memory(persona, user_id)
    short_mem = memory["short"]
    if not short_mem:
        return
    prompt = "Summarize the following conversation in 2-3 sentences:\n"
    for msg in short_mem:
        prompt += f"User: {msg['user']}\nBot: {msg['bot']}\n"
    summary = safe_api_call(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    if summary:
        memory["long"] += f"\n{summary}\n"
        memory["short"] = []
        save_memory(persona)

def add_to_user_short_memory(persona: str, user_id: str, user_msg: str, bot_reply: str):
    memory = get_user_memory(persona, user_id)
    memory["short"].append({"user": user_msg, "bot": bot_reply})
    if len(memory["short"]) > PERSONALITIES[persona]["max_short_memory"]:
        summarize_and_refresh_memory(persona, user_id)

# === Shared Chat Handler ===
async def handle_chat(ctx, persona: str, message: str):
    user_id = str(ctx.author.id)
    memory = get_user_memory(persona, user_id)
    persona_cfg = PERSONALITIES[persona]
    memory_text = memory["long"]
    recent_conv = "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in memory["short"]])
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

@bot.command()
async def chat(ctx, *, message: str):
    await handle_chat(ctx, "chat", message)

@bot.command()
async def mint(ctx, *, message: str):
    await handle_chat(ctx, "mint", message)

# === Moderation ===
def user_has_mod_role(member: discord.Member):
    if member.guild_permissions.administrator:
        return True
    return any(role.name == MOD_ROLE_NAME for role in member.roles)

async def ensure_muted_role(guild: discord.Guild) -> discord.Role:
    role = discord.utils.get(guild.roles, name="Muted")
    if role:
        return role
    try:
        role = await guild.create_role(
            name="Muted",
            reason="Mute role for moderation",
            permissions=discord.Permissions(send_messages=False, speak=False)
        )
    except Exception as e:
        print(f"[ERROR] Couldn't create Muted role: {e}")
        return None
    for ch in guild.channels:
        try:
            perms = ch.overwrites_for(role)
            if isinstance(ch, discord.TextChannel):
                perms.send_messages = False
            if isinstance(ch, discord.VoiceChannel):
                perms.speak = False
            await ch.set_permissions(role, overwrite=perms)
        except Exception as e:
            print(f"[WARN] Could not set perms in {ch.name}: {e}")
    return role

def parse_time(time_str: str) -> int:
    """Convert '10m', '2h', '3d' to seconds."""
    if time_str.isdigit():
        return int(time_str)
    unit = time_str[-1].lower()
    try:
        val = int(time_str[:-1])
    except ValueError:
        return 0
    if unit == "s":
        return val
    elif unit == "m":
        return val * 60
    elif unit == "h":
        return val * 3600
    elif unit == "d":
        return val * 86400
    return 0

@bot.command()
async def ban(ctx, member: discord.Member, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        await ctx.reply("⛔ You don't have permission.")
        return
    try:
        await member.ban(reason=reason)
        await ctx.reply(f"✅ Banned {member}. Reason: {reason}")
    except Exception as e:
        await ctx.reply("⚠️ Failed to ban.")
        print(f"[ERROR] ban: {e}")

@bot.command()
async def kick(ctx, member: discord.Member, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        await ctx.reply("⛔ You don't have permission.")
        return
    try:
        await member.kick(reason=reason)
        await ctx.reply(f"✅ Kicked {member}. Reason: {reason}")
    except Exception as e:
        await ctx.reply("⚠️ Failed to kick.")
        print(f"[ERROR] kick: {e}")

@bot.command()
async def mute(ctx, member: discord.Member, duration: str = None, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        await ctx.reply("⛔ You don't have permission.")
        return
    role = await ensure_muted_role(ctx.guild)
    if not role:
        await ctx.reply("⚠️ No Muted role.")
        return

    try:
        await member.add_roles(role, reason=reason)
        msg = f"🔇 Muted {member.mention}"
        if duration and duration[-1].lower() in ["s", "m", "h", "d"]:
            seconds = parse_time(duration)
            if seconds > 0:
                msg += f" for {duration}"
                async def unmute_after():
                    await asyncio.sleep(seconds)
                    if role in member.roles:
                        await member.remove_roles(role, reason="Auto unmute after timer")
                        try:
                            await ctx.send(f"🔊 {member.mention} has been automatically unmuted.")
                        except:
                            pass
                bot.loop.create_task(unmute_after())
        if reason:
            msg += f" | Reason: {reason}"
        await ctx.reply(msg)
    except Exception as e:
        await ctx.reply("⚠️ Failed to mute.")
        print(f"[ERROR] mute: {e}")

@bot.command()
async def unmute(ctx, member: discord.Member, *, reason: str = None):
    if not user_has_mod_role(ctx.author):
        await ctx.reply("⛔ You don't have permission.")
        return
    try:
        role = discord.utils.get(ctx.guild.roles, name="Muted")
        if role in member.roles:
            await member.remove_roles(role, reason=reason)
            await ctx.reply(f"🔊 Unmuted {member}.")
        else:
            await ctx.reply("⚠️ Member is not muted.")
    except Exception as e:
        await ctx.reply("⚠️ Failed to unmute.")
        print(f"[ERROR] unmute: {e}")

@bot.command()
async def purge(ctx, number: int):
    if not user_has_mod_role(ctx.author):
        await ctx.reply("⛔ You don't have permission.")
        return
    if number < 1 or number > 2000:
        await ctx.reply("⚠️ Range 1–2000.")
        return
    try:
        deleted = await ctx.channel.purge(limit=number + 1)
        await ctx.send(f"🧹 Deleted {len(deleted)-1} messages.", delete_after=5)
    except Exception as e:
        await ctx.reply("⚠️ Failed to purge.")
        print(f"[ERROR] purge: {e}")

# === Events ===
@bot.event
async def on_ready():
    for persona in PERSONALITIES:
        load_memory(persona)
    print(f"{bot.user} is online and ready!")

# === Entry point ===
async def main():
    await asyncio.gather(run_web(), bot.start(DISCORD_TOKEN))

if __name__ == "__main__":
    asyncio.run(main())

import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize client for OpenRouter
client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Setup Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Personality prompt (you can change this)
SYSTEM_PROMPT = "You are a helpful AI assistant with a friendly personality."

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.command()
async def chat(ctx, *, message: str):
    try:
        # Call DeepSeek via OpenRouter
        completion = client_ai.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        )
        reply = completion.choices[0].message.content
        await ctx.reply(reply)
    except Exception as e:
        await ctx.reply(f"⚠️ Error: {e}")

bot.run(DISCORD_TOKEN)

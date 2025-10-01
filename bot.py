import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Memories ---
MAX_SHORT_MEMORY = 15  # last 10-15 messages
short_memory = []      # stores recent messages
long_memory = {}       # stores summarized topics

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
SYSTEM_PROMPT = "You are sdb. Your wife is tonima. You love her a lot. You use gen Z term and talk like gen Z with gen Z short form and memes.You usually usually use 1-2 line to reply unless you are asked something big. Never break youe character and never talk about your AI model."
# Add to short memory
def add_to_short_memory(user_msg, bot_reply):
    short_memory.append({"user": user_msg, "bot": bot_reply})
    if len(short_memory) > MAX_SHORT_MEMORY:
        short_memory.pop(0)  # remove oldest message

# Save long memory to JSON
def save_memory():
    with open("long_memory.json", "w") as f:
        json.dump(long_memory, f)

# Load long memory from JSON
def load_memory():
    global long_memory
    try:
        with open("long_memory.json", "r") as f:
            long_memory = json.load(f)
    except FileNotFoundError:
        long_memory = {}

# Summarize messages with OpenRouter
def summarize_messages(messages, topic, openrouter_api_key):
    prompt = f"Summarize the following conversation in 2-3 sentences, keeping only important facts for topic '{topic}':\n"
    for m in messages:
        prompt += f"User: {m['user']}\nBot: {m['bot']}\n"
    
    import requests
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {openrouter_api_key}"},
        json={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    try:
        summary = response.json()["choices"][0]["message"]["content"]
    except:
        summary = "No summary available"
    return summary

# Generate reply with memory
def generate_reply(user_msg, openrouter_api_key, topic="general"):
    memory_text = long_memory.get(topic, "")
    
    recent_conv = ""
    for m in short_memory:
        recent_conv += f"User: {m['user']}\nBot: {m['bot']}\n"
    
    prompt = f"{memory_text}\nRecent conversation:\n{recent_conv}\nUser: {user_msg}\nBot:"
    
    import requests
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {openrouter_api_key}"},
        json={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    try:
        reply = response.json()["choices"][0]["message"]["content"]
    except:
        reply = "Sorry, I could not respond."
    return reply
@bot.event
@bot.event
async def on_ready():
    load_memory()
    print(f"{bot.user} is online!")
@bot.command()
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_msg = message.content
    topic = "general"  # you can later make this dynamic

    # Generate reply using memory
    reply = generate_reply(user_msg, OPENROUTER_API_KEY, topic)
    await message.channel.send(reply)

    # Update short-term memory
    add_to_short_memory(user_msg, reply)

    # Optional: update summary every 10 messages
    if len(short_memory) % 10 == 0:
        long_memory[topic] = summarize_messages(short_memory, topic, OPENROUTER_API_KEY)
        save_memory()
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

###################################################################################
###################################################################################
###################################################################################
#
#
import discord
from discord import app_commands
from discord.ext import commands, tasks
from discord.ext.commands import bot
import threading, asyncio
from datetime import datetime
import time
from pathlib import Path
import subprocess
import os
import yaml


def new_config():
    newconfig = {'channel_id':{'post_id':0,'heart_id':0,'average_id':0,'cursed_id':0,'nsfw_id':0},'admin':{'admin_roleid':0,'botmod_roleid':0},'main':{'enabled':False,'post_result':False}}
    return newconfig


botconfigfile = "cfg/discordbot/bot.yaml"
bottokenfile = "cfg/discordbot/TOKEN"


if botconfigfile is not None and os.path.isfile(botconfigfile):
    try:
        with open(botconfigfile, "r", encoding="utf8") as f:
            bot_config = yaml.safe_load(f)
            print("Loaded discord bot defaults successfully.")
    except (OSError, yaml.YAMLError) as e:
        print(f"Error loading defaults file {botconfigfile}:", e, file=sys.stderr)
        print("Falling back to program defaults.", file=sys.stderr)
        bot_config = new_config()
else:
    print(f"No bot config file found at {botconfigfile}, creating one.")
    with open(botconfigfile, "w", encoding="utf8") as f:
        save_config = yaml.dump(new_config(), f, default_flow_style=False)
    with open(botconfigfile, "r", encoding="utf8") as f:
        bot_config = yaml.safe_load(f)


if bottokenfile is not None and os.path.isfile(bottokenfile):
    try:
        with open(bottokenfile,'r') as file:
            bot_token = file.read().strip()
            print("Loaded Token from file.")
    except:
        print(f"Error Loading token file, please check {bottokenfile} and make sure your token is there")
        bot_config['main']['enabled'] == False
else:
    print("No TOKEN file found, Creating one.")
    with open(bottokenfile,'w') as file:
        file.write()
    print(f"Edit {bottokenfile} and paste your bot token")






intents = discord.Intents.all()
intents.members = True
intents.message_content = True

class PersistentButtons(discord.Client):
#class PersistentButtons(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix=commands.when_mentioned_or('!'), intents=intents)
        self.synced = False

    async def setup_hook(self) -> None:
        # Register the persistent view for listening here.
        # Note that this does not send the view to any message.
        # In order to do this you need to first send a message with the View, which is shown below.
        # If you have the message_id you can also pass it as a keyword argument, but for this example
        # we don't have one.
        self.add_view(Buttons())

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await bot.sync(guild = discord.Object(id = 1008397331512696893))
            self.synced = True
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')



bot = PersistentButtons()
#tree = app_commands.CommandTree(bot)

post_id = bot_config['channel_id']['post_id']
heart_id = bot_config['channel_id']['heart_id']
cursed_id = bot_config['channel_id']['cursed_id']
average_id = bot_config['channel_id']['average_id']
nsfw_id = bot_config['channel_id']['nsfw_id']
admin_roleid = bot_config['admin']['admin_roleid']
botmod_roleid = bot_config['admin']['botmod_roleid']

@bot.command(name="toggleposttest", description="Toggle post result", guild = discord.Object(id = 1008397331512696893))
async def self(interaction: discord.Interaction):
    bot_config['main']['post_result'] = not bot_config['main']['post_result']
    with open(botconfigfile, "w", encoding="utf8") as f:
        save_config = yaml.dump(bot_config, f, default_flow_style=False)
    await interaction.send_message(f"Image posting set to: {bot_config['main']['post_result']}", delete_after=10)


@bot.event
async def on_raw_reaction_add(payload):
    cursed_channel = bot.get_channel(cursed_id)
    heart_channel = bot.get_channel(heart_id)
    post_channel = bot.get_channel(post_id)
    print(f"1st trigger {payload.emoji.name}")
    if payload.channel_id != post_id:
        print("wrong channel")
        return
    do_react = False
    for r in payload.user.roles:
        if r.id == admin_roleid or botmod_roleid:
            do_react = True

    if do_react == False:
        return

    if payload.emoji.name == '🔥': # fire emoji
        msg_to_forward = await post_channel.fetch_message(payload.message_id)
        await heart_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])

    if payload.emoji.name == '😂':
        print("joy detected")
        msg_to_forward = await post_channel.fetch_message(payload.message_id)
        await cursed_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
    print("end reaction loop")




class Buttons(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)
        @discord.ui.button(label="Keeper",style=discord.ButtonStyle.gray,emoji="😍",custom_id="keeper_button") # or .primary
        async def heart_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            do_post = False
            for r in interaction.user.roles:
                if r.id == admin_roleid or botmod_roleid:
                    do_post = True
            if do_post == True:
                heart_channel = bot.get_channel(heart_id)
                msg_to_forward = interaction.message
                await heart_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                await interaction.message.delete()


        @discord.ui.button(label="Average",style=discord.ButtonStyle.gray,emoji="👌",custom_id="average_button") # or .secondary/.grey
        async def average_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            do_post = False
            for r in interaction.user.roles:
                if r.id == admin_roleid or botmod_roleid:
                    do_post = True
            if do_post == True:
                average_channel = bot.get_channel(average_id)
                msg_to_forward = interaction.message
                await average_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                await interaction.message.delete()


        @discord.ui.button(label="Cursed and Funny",style=discord.ButtonStyle.gray,emoji="🤣",custom_id="cursed_button") # or .secondary/.grey
        async def cursed_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            do_post = False
            for r in interaction.user.roles:
                if r.id == admin_roleid or botmod_roleid:
                    do_post = True
            if do_post == True:
                cursed_channel = bot.get_channel(cursed_id)
                msg_to_forward = interaction.message
                await cursed_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                await interaction.message.delete()

        @discord.ui.button(label="NSFW",style=discord.ButtonStyle.gray,emoji="🔞",custom_id="nsfw_button") # or .secondary/.grey
        async def nsfw_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            do_post = False
            for r in interaction.user.roles:
                if r.id == admin_roleid or botmod_roleid:
                    do_post = True
            if do_post == True:
                nsfw_channel = bot.get_channel(nsfw_id)
                msg_to_forward = interaction.message
                await nsfw_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                await interaction.message.delete()


        @discord.ui.button(label="Delete",style=discord.ButtonStyle.blurple,emoji="❌",custom_id="delete_button") # or .secondary/.grey
        async def delete_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            do_post = False
            for r in interaction.user.roles:
                if r.id == admin_roleid or botmod_roleid:
                    do_post = True
            if do_post == True:
                await interaction.message.delete()

@bot.command(pass_context=True)
async def togglepost(ctx):
    do_post = False
    for r in ctx.user.roles:
        if r.id == admin_roleid or botmod_roleid:
            do_post = True
    if do_post == True:
        bot_config['main']['post_result'] = not bot_config['main']['post_result']
        with open(botconfigfile, "w", encoding="utf8") as f:
            save_config = yaml.dump(bot_config, f, default_flow_style=False)
        await ctx.send(f"Image posting set to: {bot_config['main']['post_result']}", delete_after=10)
    await ctx.message.delete()
    


@bot.command(pass_context=True)
async def about(ctx):
    await ctx.message.delete()
    channel = bot.get_channel(post_id)
    await channel.send(f"Stable Diffusion Image Poster\nCreated by Vetchems\n\nAutomatically posts newly generated images (if any) from Stable Diffusion every minute", delete_after=10)


@bot.command(pass_context=True)
async def die(ctx):
    await ctx.message.delete()
    await bot.close()



async def image_send(filename,prompt,seed,subseed,seedvar,sampler_name,steps,cfg_scale,width,height,modelhash):
        channel = bot.get_channel(post_id)
        view=Buttons()
        print(f"Uploading: {filename}")
        await channel.send(f"Prompt: `{prompt}`\nSeed: `{seed}` - Subseed: `{subseed}` - Var Amount: `{seedvar}`\nSampler: `{sampler_name}` - Steps: `{steps}` - CFG Scale: `{cfg_scale}`\nImage Dimensions: `{width}x{height}`\nModel Hash: `{modelhash}`", file=discord.File(filename), view=view)

##########################
#save_sample part
def post_result(image,prompt,seed,subseed,seedvar,sampler_name,steps,cfg_scale,width,height,modelhash):
    if bot_config['main']['post_result'] == True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot.loop.create_task(image_send(image,prompt,seed,subseed,seedvar,sampler_name,steps,cfg_scale,width,height,modelhash))
        loop.close()
######################
#
def bot_start():
    bot.run(bot_token)


class BotLauncher(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.name = 'Gradio Server Thread'
        #self.demo = demo

    def run(self):
        bot.run(bot_token)
        
    def stop(self):
        self.close() # this tends to hang

if bot_config['main']['enabled'] == True:
    bot_thread = threading.Thread(target=bot_start)
    bot_thread.start()
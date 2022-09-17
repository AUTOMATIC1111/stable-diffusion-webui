###################################################################################
###################################################################################
###################################################################################
#
#
import discord
from discord.ext import commands, tasks
from discord.ext.commands import bot
import threading, asyncio
from datetime import datetime
import time
from pathlib import Path
import subprocess
import os
import yaml

botconfigfile = "cfg/discordbot/bot.yaml"
with open('cfg/discordbot/TOKEN','r') as file:
    bot_token = file.read().strip()

if botconfigfile is not None and os.path.isfile(botconfigfile):
    try:
        with open(botconfigfile, "r", encoding="utf8") as f:
            bot_config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error loading defaults file {botconfigfile}:", e, file=sys.stderr)
        print("Falling back to program defaults.", file=sys.stderr)
        bot_config = {}
else:
    bot_config = {}



def remove_line(fileName,lineToSkip):
    """ Removes a given line from a file """
    with open(fileName,'r') as read_file:
        lines = read_file.readlines()

    currentLine = 1
    with open(fileName,'w') as write_file:
        for line in lines:
            if currentLine == lineToSkip:
                pass
            else:
                write_file.write(line)
    
            currentLine += 1




intents = discord.Intents.all()
intents.members = True
intents.message_content = True
#bot = commands.Bot(command_prefix='!', intents=intents)
class PersistentButtons(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix=commands.when_mentioned_or('!'), intents=intents)

    async def setup_hook(self) -> None:
        # Register the persistent view for listening here.
        # Note that this does not send the view to any message.
        # In order to do this you need to first send a message with the View, which is shown below.
        # If you have the message_id you can also pass it as a keyword argument, but for this example
        # we don't have one.
        self.add_view(Buttons())

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')
    #    slow_count.start()


bot = PersistentButtons()

post_id = bot_config['channel_id']['post_id']
heart_id = bot_config['channel_id']['heart_id']
cursed_id = bot_config['channel_id']['cursed_id']
nsfw_id = bot_config['channel_id']['nsfw_id']
admin_roleid = bot_config['admin']['admin_roleid']

@bot.event
async def on_raw_reaction_add(payload):
    #ChID = 1015686897198170252 # #sd-img-spam
    #ChID_fire = 1015234649192149023 # #stable-diffussion
    #ChID_joy = 1015615431899369512 # #cursed-gens
    #spamchannel = bot.get_channel(ChID)
    cursed_channel = bot.get_channel(cursed_id)
    heart_channel = bot.get_channel(heart_id)
    post_channel = bot.get_channel(post_id)
    print(f"1st trigger {payload.emoji.name}")
    if payload.channel_id != post_id:
        print("wrong channel")
        return
    if payload.user_id != 205698404070850560: #Vetchems#1060
        return
    if payload.emoji.name == '🔥':
        print("fire detected")
        #giveaway_msg_channel = bot.get_channel(fb_giveaway_channel_ID)
        msg_to_forward = await post_channel.fetch_message(payload.message_id)
        await heart_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])

        #attached = await msg_to_forward.attachments[0].to_file()
        #await firechannel.send(msg_to_forward.content, file=attached)
    if payload.emoji.name == '😂':
        print("joy detected")
        msg_to_forward = await post_channel.fetch_message(payload.message_id)
        #attached = await msg_to_forward.attachments[0].to_file()
        await cursed_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
    print("end reaction loop")




class Buttons(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)
        @discord.ui.button(label="Keeper",style=discord.ButtonStyle.gray,emoji="😍",custom_id="keeper_button") # or .primary
        async def heart_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            for r in interaction.user.roles:
                if r.id == admin_roleid:
                    heart_channel = bot.get_channel(heart_id)
                    msg_to_forward = interaction.message
                    await heart_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                    await interaction.message.delete()


        @discord.ui.button(label="Average",style=discord.ButtonStyle.blurple,emoji="👌",custom_id="average_button") # or .secondary/.grey
        async def average_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            for r in interaction.user.roles:
                if r.id == admin_roleid:
                    average_channel = bot.get_channel(average_id)
                    msg_to_forward = interaction.message
                    await average_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                    await interaction.message.delete()


        @discord.ui.button(label="Cursed and Funny",style=discord.ButtonStyle.gray,emoji="🤣",custom_id="cursed_button") # or .secondary/.grey
        async def cursed_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            for r in interaction.user.roles:
                if r.id == admin_roleid:
                    cursed_channel = bot.get_channel(cursed_id)
                    msg_to_forward = interaction.message
                    await cursed_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                    await interaction.message.delete()

        @discord.ui.button(label="NSFW",style=discord.ButtonStyle.gray,emoji="🔞",custom_id="nsfw_button") # or .secondary/.grey
        async def nsfw_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            for r in interaction.user.roles:
                if r.id == admin_roleid:
                    nsfw_channel = bot.get_channel(nsfw_id)
                    msg_to_forward = interaction.message
                    await nsfw_channel.send(msg_to_forward.content, files=[await a.to_file() for a in msg_to_forward.attachments])
                    await interaction.message.delete()


        @discord.ui.button(label="Delete",style=discord.ButtonStyle.blurple,emoji="❌",custom_id="delete_button") # or .secondary/.grey
        async def deletee_button(self,interaction:discord.Interaction,button:discord.ui.Button):
            for r in interaction.user.roles:
                if r.id == admin_roleid:
                    await interaction.message.delete()

#@bot.command()
#async def button(ctx):
#    view=Buttons()
#    view.add_item(discord.ui.Button(label="URL Button",style=discord.ButtonStyle.link,url="https://github.com/lykn",emoji="<a:kannaWink:909791444661850132>"))
#    await ctx.send("This message has buttons!",view=view)


@bot.command(pass_context=True)
async def about(ctx):
    await ctx.message.delete()
    channel = bot.get_channel(post_id)
    await channel.send(f"Stable Diffusion Image Poster\nCreated by Vetchems\n\nAutomatically posts newly generated images (if any) from Stable Diffusion every minute", delete_after=10)


@bot.command(pass_context=True)
async def die(ctx):
    await ctx.message.delete()
    await bot.close()

async def image_send(filename,prompt,seed,sampler_name,steps,cfg_scale,width,height):
        channel = bot.get_channel(post_id)
        view=Buttons()
        #loop.create_task(image_send(filename_i,prompts[i],seeds[i],sampler_name,steps,cfg_scale,width,height))
        print(f"Uploading: {filename}")
        await channel.send(f"Prompt: `{prompt}`\nSeed: `{seed}`\nSampler: `{sampler_name}` - Steps: `{steps}` - CFG Scale: `{cfg_scale}`\nImage Dimensions: `{width}x{height}`", file=discord.File(filename), view=view)
        #await channel.send(file=discord.File(filename), view=view)

##########################
#save_sample part
def post_result(image,prompt,seed,sampler_name,steps,cfg_scale,width,height):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    #image = os.path.join("../", image)
    #image = join("../" + image)
    bot.loop.create_task(image_send(image,prompt,seed,sampler_name,steps,cfg_scale,width,height))
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


bot_thread = threading.Thread(target=bot_start)
bot_thread.start()
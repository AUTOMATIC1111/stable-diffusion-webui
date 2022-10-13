import discord
from discord.ext import commands, tasks
from discord.ext.commands import bot
from discord import app_commands
import threading, asyncio
from datetime import datetime
import time
from pathlib import Path
import subprocess
import os
import yaml


def new_config():
    newconfig = {'channel_id':{'server_id':0,'post_id':0,'heart_id':0,'average_id':0,'cursed_id':0,'nsfw_id':0},'admin':{'admin_roleid':0,'botmod_roleid':0},'main':{'enabled':False,'post_result':False,'post_prompt':True,'mod_buttons':True}}
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
        bot_config['main']['enabled'] = False
else:
    print("No TOKEN file found, Creating one.")
    with open(bottokenfile,'w') as file:
        file.write()
    print(f"Edit {bottokenfile} and paste your bot token")






intents = discord.Intents.all()
intents.members = True
intents.message_content = True

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
        await self.tree.sync(guild = discord.Object(id = bot_config['channel_id']['server_id']))
        print(f"Synced slash commands for {self.user}.")

    async def on_command_error(self, ctx, error):
        await ctx.reply(error, ephemeral = True)
    async def on_ready(self):
        await self.wait_until_ready()
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')


bot = PersistentButtons()

post_id = bot_config['channel_id']['post_id']
heart_id = bot_config['channel_id']['heart_id']
cursed_id = bot_config['channel_id']['cursed_id']
average_id = bot_config['channel_id']['average_id']
nsfw_id = bot_config['channel_id']['nsfw_id']
admin_roleid = bot_config['admin']['admin_roleid']
botmod_roleid = bot_config['admin']['botmod_roleid']

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
        if r.id == admin_roleid or r.id == botmod_roleid:
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
                if r.id == admin_roleid or r.id == botmod_roleid:
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
                if r.id == admin_roleid or r.id == botmod_roleid:
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
                if r.id == admin_roleid or r.id == botmod_roleid:
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
                if r.id == admin_roleid or r.id == botmod_roleid:
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
                if r.id == admin_roleid or r.id == botmod_roleid:
                    do_post = True
            if do_post == True:
                await interaction.message.delete()


async def update_bot_config():
    with open(botconfigfile, "w", encoding="utf8") as f:
        save_config = yaml.dump(bot_config, f, default_flow_style=False)

@bot.hybrid_command(name = "postresult", with_app_command = True, description = "Toggles rusulting image posting.")
@app_commands.guilds(discord.Object(id = bot_config['channel_id']['server_id']))
@commands.has_any_role(bot_config['admin']['admin_roleid'], bot_config['admin']['botmod_roleid'])
async def postresult(ctx: commands.Context):
    await ctx.defer(ephemeral = True)
    bot_config['main']['post_result'] = not bot_config['main']['post_result']
    with open(botconfigfile, "w", encoding="utf8") as f:
        save_config = yaml.dump(bot_config, f, default_flow_style=False)
    await ctx.reply(f"Image posting set to: {bot_config['main']['post_result']}", delete_after=10)
    return


@bot.hybrid_command(name = "sdhelp", with_app_command = True, description = "Displays bot help.")
@app_commands.guilds(discord.Object(id = bot_config['channel_id']['server_id']))
@commands.has_any_role(bot_config['admin']['admin_roleid'], bot_config['admin']['botmod_roleid'])
async def sdhelp(ctx: commands.Context, arg1 = None):
    await ctx.defer(ephemeral = True)
    if arg1 == None:
        await ctx.reply(":: Supported Commands ::\n\n/postresult\n/postprompt\n/modbuttons\n/setid\n/about\n/die\n\nUse /sdhelp <command> for more info.", delete_after=10)
    elif arg1 == "!postresult" or arg1 == "postresult":
        await ctx.reply("/postresult :: Toggles posting of imageresults to discord.", delete_after=10)
    elif arg1 == "!postprompt" or arg1 == "postprompt":
        await ctx.reply("/postprompt :: Toggles posting prompt and settings used for the generation.", delete_after=10)
    elif arg1 == "!modbuttons" or arg1 == "modbuttons":
        await ctx.reply("/modbuttons :: Toggles showing moderation buttons with posted results.", delete_after=10)
    elif arg1 == "!setid" or arg1 == "setid":
        await ctx.reply("/setid :: Usage !setid <post/keep/average/cursed/nsfw> <discord channel id>", delete_after=10)
    else:
        await ctx.reply(f"Unknown command: {arg1}", delete_after=10)


@bot.hybrid_command(name = "postprompt", with_app_command = True, description = "Toggles image result posting..")
@app_commands.guilds(discord.Object(id = bot_config['channel_id']['server_id']))
@commands.has_any_role(bot_config['admin']['admin_roleid'], bot_config['admin']['botmod_roleid'])
async def postprompt(ctx: commands.Context):
    await ctx.defer(ephemeral = True)
    bot_config['main']['post_prompt'] = not bot_config['main']['post_prompt']
    with open(botconfigfile, "w", encoding="utf8") as f:
        save_config = yaml.dump(bot_config, f, default_flow_style=False)
    await ctx.reply(f"Prompt posting set to: {bot_config['main']['post_prompt']}", delete_after=10)
    return

@bot.hybrid_command(name = "modbuttons", with_app_command = True, description = "Toggles showing of moderation buttons when posting image result.")
@app_commands.guilds(discord.Object(id = bot_config['channel_id']['server_id']))
@commands.has_any_role(bot_config['admin']['admin_roleid'], bot_config['admin']['botmod_roleid'])
async def modbuttons(ctx: commands.Context):
    await ctx.defer(ephemeral = True)
    bot_config['main']['mod_buttons'] = not bot_config['main']['mod_buttons']
    with open(botconfigfile, "w", encoding="utf8") as f:
        save_config = yaml.dump(bot_config, f, default_flow_style=False)
    await ctx.reply(f"Show moderation buttons set to: {bot_config['main']['mod_buttons']}", delete_after=10)
    return

@bot.hybrid_command(name = "setid", with_app_command = True, description = "Sets channel and role id's.")
@app_commands.guilds(discord.Object(id = bot_config['channel_id']['server_id']))
@commands.has_any_role(bot_config['admin']['admin_roleid'], bot_config['admin']['botmod_roleid'])
async def setid(ctx: commands.Context, arg1 = None, arg2: int = None):
    await ctx.defer(ephemeral = True)
    if arg1 != None and arg2 != None:
        if arg1 == "post": 
            bot_config['channel_id']['post_id'] = arg2
            post_id = arg2
            await ctx.reply(f"Post channel ID set to: {arg2}", delete_after=10)
            await update_bot_config()
            return
        elif arg1 == "keep": 
            bot_config['channel_id']['heart_id'] = arg2
            heart_id = arg2
            await ctx.reply(f"Keeper channel ID set to: {arg2}", delete_after=10)
            await update_bot_config()
            return
        elif arg1 == "average": 
            bot_config['channel_id']['average_id'] = arg2
            average_id = arg2
            await ctx.reply(f"Average channel ID set to: {arg2}", delete_after=10)
            await update_bot_config()
            return
        elif arg1 == "cursed": 
            bot_config['channel_id']['cursed_id'] = arg2
            cursed_id = arg2
            await ctx.reply(f"Cursed channel ID set to: {arg2}", delete_after=10)
            await update_bot_config()
            return
        elif arg1 == "nsfw": 
            bot_config['channel_id']['nsfw_id'] = arg2
            nsfw_id = arg2
            await ctx.reply(f"NSFW channel ID set to: {arg2}", delete_after=10)
            await update_bot_config()
            return
        else:
            await ctx.reply(f"Invalid argument: {arg1}", delete_after=10)
            return
    else:
        await ctx.reply(f"Missing arguments, use: /setid <name> <id>\nValid names are: post, keep, average, cursed, nsfw", delete_after=10)
        return


@bot.hybrid_command(name = "about", with_app_command = True, description = "Shows info about the bot.")
@app_commands.guilds(discord.Object(id = bot_config['channel_id']['server_id']))
async def about(ctx: commands.Context):
    await ctx.defer(ephemeral = True)
    await ctx.reply(f"Stable Diffusion Image Poster\nCreated by Vetchems\n\nAutomatically posts newly generated images generated by Stable Diffusions Web UI.\n\nhttps://github.com/Vetchems/stable-diffusion-discord-vetch", delete_after=10)


@bot.hybrid_command(name = "die", with_app_command = True, description = "Force disconnect the bot (restart stable diffusion to reconnect).")
@app_commands.guilds(discord.Object(id = bot_config['channel_id']['server_id']))
@commands.has_any_role(bot_config['admin']['admin_roleid'], bot_config['admin']['botmod_roleid'])
async def die(ctx: commands.Context):
    await ctx.defer(ephemeral = True)
    await ctx.reply("Bot will now disconnect.")
    await bot.close()



async def image_send(filename,prompt,negative_prompt,seed,subseed,seedvar,sampler_name,steps,cfg_scale,width,height,modelhash,seed_resize_w, seed_resize_h):
        channel = bot.get_channel(post_id)
        view=Buttons()
        print(f"Uploading: {filename}")
        if bot_config['main']['post_prompt'] == True:
            prompt_info = prompt
    
            if negative_prompt != "":
                prompt_info = f"{prompt_info}\nNegative prompt: {negative_prompt}\nSteps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}, Size: {width}x{height}, Model hash: {modelhash}"
            else:
                prompt_info = f"{prompt_info}\nSteps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}, Size: {width}x{height}, Model hash: {modelhash}"
    
            if str(seedvar) != "0":
                prompt_info = f"{prompt_info}, Variation seed: {subseed}, Variation seed strength: {seedvar}"
    
            if seed_resize_w != 0 and seed_resize_h != 0:
                prompt_info = f"{prompt_info}, Seed resize from: {seed_resize_w}x{seed_resize_h}"
        else:
            prompt_info = ""

        if bot_config['main']['mod_buttons']:
            await channel.send(prompt_info, file=discord.File(filename), view=view)
        else:
            await channel.send(prompt_info, file=discord.File(filename))

##########################
#save_sample part
def post_result(image,prompt,negative_prompt,seed,subseed,seedvar,sampler_name,steps,cfg_scale,width,height,modelhash,seed_resize_w, seed_resize_h):
    if bot_config['main']['post_result'] == True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot.loop.create_task(image_send(image,prompt,negative_prompt,seed,subseed,seedvar,sampler_name,steps,cfg_scale,width,height,modelhash,seed_resize_w,seed_resize_h))
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
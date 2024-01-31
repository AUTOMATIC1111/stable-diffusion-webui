/*
 * @Author: SpenserCai
 * @Date: 2023-08-10 10:05:12
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-10 16:49:50
 * @Description: file content
 */
package core

import (
	"DeoldifyBot/global"
	"log"

	"github.com/bwmarrin/discordgo"
)

func Ready(s *discordgo.Session, event *discordgo.Ready) {
	log.Printf("Logged in as: %v#%v", s.State.User.Username, s.State.User.Discriminator)
}

func InteractionCreate(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if h, ok := commandHandlers[i.ApplicationCommandData().Name]; ok {
		h(s, i)
	}

}

func AddCommand() {
	log.Println("Adding commands...")
	global.AppCommand = make([]*discordgo.ApplicationCommand, len(commandList))
	for i, v := range commandList {
		cmd, err := global.DiscordSession.ApplicationCommandCreate(global.DiscordSession.State.User.ID, "1138404797364580415", v)
		if err != nil {
			log.Panicf("Cannot create '%v' command: %v", v.Name, err)
		}
		global.AppCommand[i] = cmd
	}
}

func RemoveCommand() {
	log.Println("Removing commands...")

	for _, v := range global.AppCommand {
		err := global.DiscordSession.ApplicationCommandDelete(global.DiscordSession.State.User.ID, "1138404797364580415", v.ID)
		if err != nil {
			log.Panicf("Cannot delete '%v' command: %v", v.Name, err)
		}
	}
}

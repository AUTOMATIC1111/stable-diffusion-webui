/*
 * @Author: SpenserCai
 * @Date: 2023-08-10 09:52:57
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-10 15:42:44
 * @Description: file content
 */
package core

import (
	CmdHandlers "DeoldifyBot/core/command_handlers"

	"github.com/bwmarrin/discordgo"
)

var (
	commandHandlers = map[string]func(*discordgo.Session, *discordgo.InteractionCreate){
		"deoldify": CmdHandlers.DeoldifyCommandHandler,
	}
)

var (
	renderFactorMin = 1.0
	renderFactorMax = 50.0
	commandList     = []*discordgo.ApplicationCommand{
		{
			Name:        "deoldify",
			Description: "Deoldify a image",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "image_url",
					Description: "The url of the image",
					Required:    true,
				},
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "render_factor",
					Description: "The render factor of the image",
					Required:    false,
					MinValue:    &renderFactorMin,
					MaxValue:    float64(renderFactorMax),
				},
				{
					Type:        discordgo.ApplicationCommandOptionBoolean,
					Name:        "artistic",
					Description: "Whether to use artistic mode",
					Required:    false,
				},
			},
		},
	}
)

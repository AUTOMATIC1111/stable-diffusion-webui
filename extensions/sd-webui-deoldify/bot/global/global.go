/*
 * @Author: SpenserCai
 * @Date: 2023-08-09 22:53:14
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-12 21:41:01
 * @Description: file content
 */
package global

import (
	Common "DeoldifyBot/common"
	"DeoldifyBot/config"

	SdClient "github.com/SpenserCai/sd-webui-go"
	"github.com/bwmarrin/discordgo"
)

var (
	Config         *config.Config
	DiscordSession *discordgo.Session
	AppCommand     []*discordgo.ApplicationCommand
	StableClient   *SdClient.StableDiffInterface
	UsingCount     int64               = 0
	ActionQueue    *Common.ActionQueue = Common.NewActionQueue(5)
)

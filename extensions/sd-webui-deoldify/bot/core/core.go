/*
 * @Author: SpenserCai
 * @Date: 2023-08-09 23:31:50
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-10 23:57:19
 * @Description: file content
 */
package core

import (
	"DeoldifyBot/global"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/bwmarrin/discordgo"
)

func RunBot() {
	discord, err := discordgo.New("Bot " + global.Config.Token)
	if err != nil {
		fmt.Println(err)
		return
	}
	global.DiscordSession = discord
	global.DiscordSession.AddHandler(Ready)
	global.DiscordSession.AddHandler(InteractionCreate)
	go global.ActionQueue.Run()
	err = global.DiscordSession.Open()
	if err != nil {
		fmt.Println(err)
		return
	}
	AddCommand()
	defer global.DiscordSession.Close()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	log.Println("Press Ctrl+C to exit")
	<-stop
	RemoveCommand()
	log.Println("Gracefully shutting down.")
}

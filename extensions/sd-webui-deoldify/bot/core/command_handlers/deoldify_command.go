/*
 * @Author: SpenserCai
 * @Date: 2023-08-10 15:37:59
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-11 00:12:50
 * @Description: file content
 */
package command_handlers

import (
	Global "DeoldifyBot/global"
	Service "DeoldifyBot/service"
	"bytes"
	"encoding/base64"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/bwmarrin/discordgo"
)

type Options struct {
	ImageURL     string `json:"image_url"`
	RenderFactor int64  `json:"render_factor"`
	Artistic     bool   `json:"artistic"`
}

func SetOptions(disOps map[string]*discordgo.ApplicationCommandInteractionDataOption, opt *Options) {
	if tmp_opt, ok := disOps["image_url"]; ok {
		opt.ImageURL = tmp_opt.StringValue()
	}
	if tmp_opt, ok := disOps["render_factor"]; ok {
		opt.RenderFactor = tmp_opt.IntValue()
	}
	if opt.RenderFactor < 1 {
		opt.RenderFactor = 35
	}
	if tmp_opt, ok := disOps["artistic"]; ok {
		opt.Artistic = tmp_opt.BoolValue()
	}

}

func SendInfoMsg(s *discordgo.Session, i *discordgo.InteractionCreate, opt *Options) {
	msgOut := "Render Factor: " + strconv.FormatInt(opt.RenderFactor, 10) + "\n" +
		"Artistic: " + strconv.FormatBool(opt.Artistic) + "\n" +
		"Image: " + opt.ImageURL
	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: msgOut,
		},
	})
}

func Action(s *discordgo.Session, i *discordgo.InteractionCreate, opt *Options) {
	msg, err := s.FollowupMessageCreate(i.Interaction, true, &discordgo.WebhookParams{
		Content: "Deoldifing...",
		Files:   []*discordgo.File{},
	})
	if err != nil {
		s.FollowupMessageCreate(i.Interaction, true, &discordgo.WebhookParams{
			Content: "Something went wrong",
		})
		return
	}
	imageB64, err := Service.Deoldify(opt.ImageURL, opt.RenderFactor, opt.Artistic)
	if err != nil {
		errMsg := fmt.Sprintf("Error: %v", err)
		s.FollowupMessageEdit(i.Interaction, msg.ID, &discordgo.WebhookEdit{
			Content: &errMsg,
		})
	} else {
		// 文件名字当前年月日时分秒.png, 例如 20210810130405.png
		fileName := time.Now().Format("20060102150405") + ".png"
		reader, deErr := base64.StdEncoding.DecodeString(imageB64)
		if deErr != nil {
			log.Println(deErr)
		}
		file := &discordgo.File{
			Name:        fileName,
			Reader:      bytes.NewReader(reader),
			ContentType: "image/png",
		}
		successMsg := "Success!"
		_, err := s.FollowupMessageEdit(i.Interaction, msg.ID, &discordgo.WebhookEdit{
			Content: &successMsg,
			Files:   []*discordgo.File{file},
		})
		if err != nil {
			log.Println(err)
		}
	}
}

func DeoldifyCommandHandler(s *discordgo.Session, i *discordgo.InteractionCreate) {
	opt := Options{}
	options := i.ApplicationCommandData().Options
	optionMap := make(map[string]*discordgo.ApplicationCommandInteractionDataOption, len(options))
	for _, opt := range options {
		optionMap[opt.Name] = opt
	}
	SetOptions(optionMap, &opt)
	SendInfoMsg(s, i, &opt)
	Global.ActionQueue.AddTask(func() { Action(s, i, &opt) })

}

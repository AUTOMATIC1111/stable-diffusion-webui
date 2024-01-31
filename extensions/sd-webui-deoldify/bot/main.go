/*
 * @Author: SpenserCai
 * @Date: 2023-08-09 22:48:29
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-12 21:30:23
 * @Description: file content
 */
package main

import (
	Global "DeoldifyBot/global"
	"encoding/json"
	"os"
	"path/filepath"

	SdClient "github.com/SpenserCai/sd-webui-go"

	Core "DeoldifyBot/core"
)

func LoadConfig() error {
	// 从程序运行目录下的 config.json 文件中读取配置
	exePath, err := os.Executable()
	if err != nil {
		return err
	}
	exeDir := filepath.Dir(exePath)
	configPath := filepath.Join(exeDir, "config.json")
	file, err := os.Open(configPath)
	if err != nil {
		return err
	}
	defer file.Close()
	err = json.NewDecoder(file).Decode(&Global.Config)
	if err != nil {
		return err
	}
	Global.StableClient = SdClient.NewStableDiffInterface(Global.Config.SdWebUiAddr)
	return nil

}

func main() {
	LoadConfig()
	Core.RunBot()

}

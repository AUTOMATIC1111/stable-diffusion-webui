/*
 * @Author: SpenserCai
 * @Date: 2023-08-10 15:31:55
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-12 21:41:43
 * @Description: file content
 */
package service

import (
	Global "DeoldifyBot/global"

	"github.com/SpenserCai/sd-webui-go/intersvc"
)

func Deoldify(imageURL string, renderFactor int64, artistic bool) (string, error) {
	deoldify_inter := &intersvc.DeoldifyImage{
		RequestItem: &intersvc.DeoldifyImageRequest{
			InputImage:   imageURL,
			RenderFactor: &renderFactor,
			Artistic:     &artistic,
		},
	}
	deoldify_inter.Action(Global.StableClient)
	if deoldify_inter.Error != nil {
		return "", deoldify_inter.Error
	}
	response := deoldify_inter.GetResponse()
	return response.Image, nil

}

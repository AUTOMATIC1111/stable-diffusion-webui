/*
 * @Author: SpenserCai
 * @Date: 2023-08-10 23:10:29
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-11 00:38:38
 * @Description: file content
 */
package common

import (
	"fmt"
	"log"
	"time"
)

// 一个执行队列
type ActionQueue struct {
	// 最大并发数
	MaxConcurrent int
	// 当前并发数
	CurrentConcurrent int
	// 任务队列
	TaskQueue chan func()
}

// 创建一个执行队列
func NewActionQueue(maxConcurrent int) *ActionQueue {
	return &ActionQueue{
		MaxConcurrent:     maxConcurrent,
		CurrentConcurrent: 0,
		TaskQueue:         make(chan func(), 100),
	}
}

// 向执行队列中添加一个任务
func (aq *ActionQueue) AddTask(task func()) {
	log.Println("AddTask")
	aq.TaskQueue <- task
	log.Println("AddTask success")
}

// 执行队列中的任务
func (aq *ActionQueue) Run() {
	for {
		if aq.CurrentConcurrent < aq.MaxConcurrent && len(aq.TaskQueue) > 0 {
			aq.CurrentConcurrent++
			// 当前并发数，当前任务队列长度
			logOut := fmt.Sprintf("CurrentConcurrent: %d, TaskQueueLen: %d", aq.CurrentConcurrent, len(aq.TaskQueue))
			log.Println(logOut)
			go func() {
				task := <-aq.TaskQueue
				task()
				aq.CurrentConcurrent--
			}()
		}
		// 防止CPU占用过高
		time.Sleep(time.Millisecond * 10)
	}
}

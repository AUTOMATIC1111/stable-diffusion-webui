| 术语               | 直译     | 翻译             | 说明                                                         |
| ------------------ | -------- | ---------------- | ------------------------------------------------------------ |
|界面相关| - | - | - |
| stable diffusion   | 稳定扩散 | stable diffusion | 一个开源的文本到图像 latent diffusion 模型，能够在给定任何文本输入的情况下生成相关的图像 |
| SD                 |          |                  | stable diffusion 的缩写                                      |
| webUI              | 网络用户界面 | Webui | 本意指依赖浏览器设计的网络用户界面，当前特指 “AUTOMATIC1111’s stable-diffusion-webui”（[stable diffusion 网页用户界面](https://github.com/AUTOMATIC1111/stable-diffusion-webui)），由AUTOMATIC1111开发 |
| naifu              |          |                  | 一种模仿 Novel AI 官方页面的用户界面|
| t2i                |          | 文生图           | txt to img 的缩写                                            |
| i2i                |          | 图生图           | img to img 的缩写                                            |
| Prompt             |          | 提示词           | 原意指以文本或其他形式展现的操作提示或用户对程序的提示性输入内容，此处专指作为 stable diffusion 算法的文本内容，通常以一段文本的形式出现，可以是自然语言，也可以是由 Tag 组成的一段内容|
| Tag                |          | Tag              | 专指以逗号分隔、具有特定含义的一系列单词或词组，其具体释义应参见 [Danbooru Wiki](https://danbooru.donmai.us/wiki_pages/help:home)<br>某些教程会使用 Tag 作为 Prompt 的翻译甚至混淆二者的概念，但 Tag 只是 prompt 的一种形式，且实际上只有部分模型（基于NAI泄露的模型训练或包含此模型文本编码器的融合模型）可以以 tag 作为提示词输入，大多数模型只接受以自然语言作为提示词输入，无法正确响应 Tag 的特定含义 |
| token              |          | 词元             | prompt 被 CLIP 处理后得到的一系列数字标签，是 prompt 的基础计数单位和最小语义单位，注意，Tag 与 token 之间没有任何直接的数量关系|
| CFG Scale          | CFG指数  | 提示词相关性     | Classifier Free Guidance Scale 的缩写，基于实际功能翻译。    |
| Denoising strength | 去噪强度 | 重绘幅度         | 这里的“去噪”其实并非常规的去噪（不是那种后处理滤镜），而是同时指“加噪”强度。<br>在SD进行取样的过程中，他要先有噪点然后逐渐从噪点“恢复”成图片。这个“恢复”的过程就叫去噪。而假设我们有1份噪点（100%），step则决定每个step去除多少%的噪点。比如100步就是每步1%<br>i2i的生成过程就是，假如去噪强度是1.0，那他会加100%的噪点，然后跑100%的取样。而假如去噪强度是0.0，那他就不加噪点，也就是不做取样，直接编码进潜空间然后再直接解码回来。所以很自然0.5的去噪就只会跑一半的step，因为他只加了50%的噪点。<br>其实在sd官方，这个参数不叫Denoising strength，而是直接叫`strength` - `strength for noising/unnoising`（加噪/去噪的强度），我觉得以功能取代原有名字也是一种解决办法。 |
| inpaint            | 绘画     | 局部重绘         | 在SD里这个标签对应的功能是绘制蒙版，并对蒙版区域进行局部重绘，即依赖蒙版区域外的内容作为参照对蒙版内的区域进行重绘（选择重绘蒙版外的操作仅是将蒙版反色，并非outpaint）|
| outpaint           |          | 向外绘制         |与inpaint相反，是以画面内内容作为参照向画面四周区域绘制新内容|
| latent space       | 潜在空间  | 潜空间          |Latent Diffusion 算法的核心，为减小计算量将像素内容压缩映射产生的数据空间|
| latent noise       | 潜在噪声  | 潜空间噪声       |                                                              |
| latent zeroes      | 潜在零    | 潜空间数值零     |                                                              |
| image grids        | 图像网格  | 宫格图           |                                                              |
| Slerp              |          | 球面线性插值     |                                                              |
| extension          | 扩展     | 插件/扩展/拓展    | 为避免混淆，统一将可显示在`extension`页面的所有第三与内置方功能模块称为`扩展`，并将`插件`与`拓展`作为其别名，表示同一个概念 |
| Aesthetic          |          | 美术风格         |                                                              |
| tokenizer          |          | 词元分析器       | 
| | | | -|                                                             |
|模型相关| - | - | - |
| checkpoint         | 检查点   | ckpt             | 专门指代 stable diffusion 训练过程中模型的检查点，又因其可以直接作为 stable diffusion 推理所使用的模型，故也被叫做 `stable diffusion 模型` 或者 `SD模型`，通常以 `ckpt` 作为其文件扩展名，故亦可简称为 `ckpt`，后出于安全因素转变为多以 `safetensors` 格式储存 |
| CLIP               |          |                  | Contrastive Language-Image Pre-Training的缩写。CLIP是用于处理文本理解文本的神经网络，它把你的自然语言提示语分割成词元并转化为向量供 diffusion 模型使用，是一个完整 `ckpt` 应具有的一个数据模块 |
| VAE                | 变分自编码器 | VAE          |VAE 是用于 RGB图像 与 潜空间图像 间相互转化的一种神经网络，是一个完整 `ckpt` 应具有的一个数据模块|
| U-net              | U型网络  | Unet             |此处 Unet 特指 `ckpt` 中用于处理潜空间数据的网络结构，因其示意图形状酷似字母U而得名，是一个完整 `ckpt` 应具有的一个数据模块|
| Refiner            | 精炼器   |  Refiner          |原特指 SDXL 中新添加的一个 Unet 类型，专门用于在采样的最后几步为图像添加细节，现亦指使用其他模型（版本相同）的 unet 在生成的最后几步替换原先的 unet 以达到细化画面的作用|
| NAI                |          |                  | Novel AI，同时也指代2022年10月 Novel AI 泄露的一批 `ckpt`                                    |
| WD                 |          |                  | waifu diffusion的缩写       |
| | | | -|
|采样器相关| - | - | - |
| ETA                | η        | η值              |原参数名为 eta，现用其代表的希腊字母代称该参数及相关概念，该参数物理意义为η噪声倍率，非0的η噪声倍率会导致采样器随步数增加结果发生变化|
| Sigma              | σ        | σ值              |原参数名为 sigma，现用其代表的希腊字母代称该参数及相关概念，该参数物理意义为σ噪声倍率，用于抵消采样期间细节损失的额外噪声量|
| Rho                | ρ        | ρ值              |原参数名为 rho，现用其代表的希腊字母代称该参数及相关概念，该参数物理意义为描述噪声调度器噪声下降速率|
| ENSD               |          |η 噪声种子 偏移量| Eta noise seed delta                                                             |
| | | | -|
|附加网络相关| - | - | - |
| Hypernetwork       |          | 超网络           |                                                              |
| embedding          | 嵌入     | 嵌入式 | 早期与 Textual Inversion 混用，虽然严格意义上二者含义并不相同，但当前都指代同一种附加网络模型，且 embedding 在早期使用频率更高，故保留 |
| Textual Inversion  | 文本反转 | 嵌入式 | [参考链接](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion)        |
| LoRA               | 低秩微调  | LoRA |  |




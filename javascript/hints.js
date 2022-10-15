// mouseover tooltips for various UI elements

titles = {
    "Sampling steps|采样步数": "迭代改进生成图像的次数；较高的值需要更长的时间；非常低的值可能会产生不良结果",
    "Sampling method|采样算法": "使用哪种算法生成图像",
	"GFPGAN": "使用GFPGAN来修复低分辨率人脸",
	"Euler a": "Euler Ancestral(欧拉始祖)-非常有创意的模型，根据采样步数，每个人都可以得到完全不同的图片，将采样步数设置为高于30-40没有帮助",
	"DDIM": "降噪扩散隐式模型-最擅长修复",

	"Batch count|批次数量": "要创建多少个批次图像（最终图像数=批次数量*单批数量）",
	"Batch size|单批数量": "单批中要创建多少图像（最终图像数=批次数量*单批数量）",
    "CFG Scale|分类自由引导系数": "分类自由引导系数-图像与提示词的一致程度-较低的值会产生更具创造性的结果",
    "Seed|随机种": "一个确定随机数生成器输出的值-如果创建一个具有相同参数的图像，并将其作为另一个图像的种子，则会得到相同的结果.(注意，生成的图像都有随机种，如果你想用也可以在这里直接输入)",
    "\u{1f3b2}\ufe0f": "将Seed随机种设置为-1，这将在每次使用时会有一个新的随机数",
    "\u267b\ufe0f": "重复使用上一次的随机种.(注意，生成的图像都有随机种，如果你想用也可以在这里直接输入)",
    "\u{1f3a8}": "添加一个随机艺术家名字到提示词",
    "\u2199\ufe0f": "从提示词获取生成参数(作者挨打不写说明，测了半小时..咳咳，功能是这样，生成了图片下面由三行参数，第一行是提示词，第二行是参数，第三行是花费时间，所以你可以把第二行参数复制到提示词那里，它会自动帮助你设置成那些参数...)",
    "\u{1f4c2}": "打开输出文件夹",

    "Inpaint a part of image": "Draw a mask over an image, and the script will regenerate the masked area with content according to prompt",
    "SD upscale": "Upscale image normally, split result into tiles, improve each tile using img2img, merge whole image back",

    "Just resize|仅调整尺寸": "将图像大小调整为目标分辨率。除非高度和宽度匹配，否则长宽比将不正确。",
    "Crop and resize|裁剪并调整尺寸": "调整图像大小，以使图像填充整个目标分辨率。突出的裁剪部分.",
    "Resize and fill|调整尺寸并填充": "调整图像大小，使整个图像在目标分辨率范围内。用图像的颜色填充空白。",

    "Mask blur|蒙板模糊": "处理前模糊遮罩的程度，以像素为单位。",
    "Masked content|遮挡内容": "决定蒙板区域如何处理(决定填补的内容类型)",
    "fill|填充": "fill it with colors of the image",
    "original|原样绘制": "保留原来的样式(尽可能)",
    "latent noise|潜在空间噪波": "用潜在空间噪波填充",
    "latent nothing潜在空间为空": "用潜在空间空白填充",
    "Inpaint at full resolution|以全分辨率填充绘制": "将蒙板区域放大到目标分辨率，进行蒙板绘画，缩小并粘贴到原始图像中",

    "Denoising strength|降噪强度": "确定算法对图像内容的保留程度。值为0时，一切都不会改变，值为1时，您将得到一个不相关的图像。如果值低于1.0，则处理所需的步骤将少于“采样步骤”滑块指定的步骤.",
    "Denoising strength change factor": "In loopback mode, on each loop the denoising strength is multiplied by this value. <1 means decreasing variety so your sequence will converge on a fixed picture. >1 means increasing variety so your sequence will become more and more chaotic.",

    "Skip": "停止处理当前图像并继续处理后续(也就是打断当前，直接处理下一个)。",
    "Interrupt|终止": "立刻停止计算图像，并展示当前已经计算好的图像。",
    "Save|保存": "将图像写入目录（default-log/images）并将生成参数写入csv文件.",

    "X values": "Separate values for X axis using commas.",
    "Y values": "Separate values for Y axis using commas.",

    "None": "不需要特殊处理",
    "Prompt matrix|提示词矩阵": "使用竖线字符（|）将提示分隔为多个部分，脚本将为每个部分的组合创建一张图片（第一部分除外，它将以所有组合出现）",
    "X/Y plot": "Create a grid where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",
    "Custom code": "Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",
    "Prompt order": "Separate a list of words with commas, and the script will make a variation of prompt with those words for their every possible order",

    "Tiling|生成平铺图像(无缝)": "生成可平铺的图像(无缝).",
    "Tile overlap": "For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",

    "Variation seed|变体随机种": "将不同图片的随机种混合到一代中.",
    "Variation strength|变体强度": "产生的变化有多大。为0时，将没有效果。在1时，您将得到变异种子的完整图片（除了祖先采样器，在那里您只会得到一些东西）。.",
    "Resize seed from height|高度调整随机种": "尝试生成与使用相同随机种以指定分辨率生成的图片类似的图片",
    "Resize seed from width|宽度调整随机种": "尝试生成与使用相同随机种以指定分辨率生成的图片类似的图片",

    "Interrogate": "Reconstruct prompt from existing image and put it into the prompt field.",

    "Images filename pattern|图像文件名样式": "使用以下标记来定义如何选择图像的文件名(注意,输入格式为[类型1],[类型2]，中文的是翻译不要输进去)：[steps]采样步数、[cfg]CFG系数,[prompt]提示词空格换为下划线和逗号,[prompt_no_styles]提示词含空格和逗号,[prompt_spaces]提示词空格,[width]宽度,[height]高度,[styles]风格,[sampler]采样器,[seed]随机种,[model_hash]模型哈希值,[prompt_words]提示词语句,[date]保存时日期,[datetime]保存时日期时间,[job_timestamp]处理完成时日期时间；默认设置为空。",
    "Directory name pattern|目录名称模式": "图像文件名样式": "使用以下标记来定义如何选择图像的文件名(注意,输入格式为[类型1],[类型2]，中文的是翻译不要输进去)：[steps]采样步数、[cfg]CFG系数,[prompt]提示词空格换为下划线和逗号,[prompt_no_styles]提示词含空格和逗号,[prompt_spaces]提示词空格,[width]宽度,[height]高度,[styles]风格,[sampler]采样器,[seed]随机种,[model_hash]模型哈希值,[prompt_words]提示词语句,[date]保存时日期,[datetime]保存时日期时间,[job_timestamp]处理完成时日期时间；默认设置为空。",
    "Max prompt words|最大提示词": "设置要在[prompt_words]选项中使用的最大字数；注意：如果单词太长，可能会超过系统可以处理的文件路径的最大长度",

    "Loopback": "Process an image, use it as an input, repeat.",
    "Loops": "How many times to repeat processing an image and using it as input for the next iteration",

    "Style 1|风格1": "要应用的样式；样式具有用于正面和负面提示的组件，并应用于两者",
    "Style 2|风格2": "要应用的样式；样式具有用于正面和负面提示的组件，并应用于两者",
    "Apply style|应用风格": "插入所选风格到提示词区",
    "Create style|创建风格": "保存当前提示词作为一个风格，如果你添加{prompt}标记到文本中，则该样式将在你应用风格时被添加到提示词区",

    "Checkpoint name|检查点名称": "在制作图像之前，从检查点加载权重。您可以使用哈希或文件名的一部分（如设置中所示）作为检查点名称。建议与Y轴一起使用，以减少切换。",

    "vram": "Torch active: Peak amount of VRAM used by Torch during generation, excluding cached data.\nTorch reserved: Peak amount of VRAM allocated by Torch, including all active and cached data.\nSys VRAM: Peak amount of VRAM allocation across all applications / total GPU VRAM (peak utilization%).",

    "Highres. fix|高分辨率修正": "默认分两步生成：1创建不完整低分辨率图像 2在不改变构图的情况下放大并改进其中的细节",
    "Scale latent|潜在空间放大": "1创建完整低分辨率图像 2放大并改善细节(XX论坛:开启和关闭的主要区别在于第一步创建的低分辨率图像是否完整)",

    "Eta noise seed delta|Eta德尔塔噪波随机种": "如果该值为非零值，则将其添加到随机种中，并用于在使用Eta采样器时初始化RNG以获取噪波。您可以使用它生成更多的图像变体，或者如果您知道自己在做什么，也可以使用它来匹配其他软件的图像。",
    "Do not add watermark to images|不向图像添加水印": "如果启用此选项，则不会将水印添加到创建的图像。警告：如果不添加水印，可能会以不道德的方式进行攻击。",

    "Filename word regex|文件名词条正则": "此正则表达式将用于从文件名中提取单词，并使用下面的选项将它们连接到用于训练的标签文本中。留空以保持文件名文本不变。",
    "Filename join string|文件名加入字符串": "如果启用上述选项，则此字符串将用于将拆分的单词连接到一行中。",

    "Quicksettings list|快速设置列表": "设置名称列表，以逗号分隔，用于应该转到顶部快速访问栏而不是通常的设置选项卡的设置。有关设置名称，请参阅modules/shared.py。需要重新启动才能应用.",
    "Weighted sum": "Result = A * (1 - M) + B * M",
    "Add difference": "Result = A + (B - C) * M",
}


onUiUpdate(function(){
	gradioApp().querySelectorAll('span, button, select, p').forEach(function(span){
		tooltip = titles[span.textContent];

		if(!tooltip){
		    tooltip = titles[span.value];
		}

		if(!tooltip){
			for (const c of span.classList) {
				if (c in titles) {
					tooltip = titles[c];
					break;
				}
			}
		}

		if(tooltip){
			span.title = tooltip;
		}
	})

	gradioApp().querySelectorAll('select').forEach(function(select){
	    if (select.onchange != null) return;

	    select.onchange = function(){
            select.title = titles[select.value] || "";
	    }
	})
})

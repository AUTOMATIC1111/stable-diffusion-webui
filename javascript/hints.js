// mouseover tooltips for various UI elements

titles = {
    "采样步数/Sampling Steps": "生成的图像需要迭代改进多少次;数值越高需要的时间越长;非常低的值会产生不好的结果/How many times to improve the generated image iteratively; higher values take longer; very low values can produce bad results",
    "采样方法/Sampling method": "使用哪种算法生成图像/Which algorithm to use to produce the image",
	"GFPGAN": "利用GFPGAN神经网络修复低质量的人脸/Restore low quality faces using GFPGAN neural network",
	"Euler a": "欧拉遗传-非常有创意,每一张都可以根据步数得到完全不同的图片,设置步数高于30-40没有用/Euler Ancestral - very creative, each can get a completely different picture depending on step count, setting steps to higher than 30-40 does not help",
	"DDIM": "去噪扩散隐式模型-最好的图像修补模型/Denoising Diffusion Implicit Models - best at inpainting",

	"生成次数/Batch count": "要生成多少次图像/How many batches of images to create",
	"每次数量/Batch size": "一次生成多少张图像/How many image to create in a single batch",
    "CFG指数/CFG Scale": "分类器引导量化值-生成的图像要多大程度符合关键词描述-值越低越能产生更有创意的结果/Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results",
    "图像生成种子/Seed": "决定随机数生成器的值-如果你创建的图像跟另一个图像具有相同的参数,关键词和随机数,你会得到同样的结果/A value that determines the output of random number generator - if you create an image with same parameters and seed as another image, you'll get the same result",
    "\u{1f3b2}\ufe0f": "将图像生成种子设置为-1,这样每次生成图像都会用一个新的随机数/Set seed to -1, which will cause a new random number to be used every time",
    "\u267b\ufe0f": "重复使用上一个图像生成种子,通常在它是随机的情况下是有用的/Reuse seed from last generation, mostly useful if it was randomed",
    "\u{1f3a8}": "随机将一个艺术家添加到关键词语句中/Add a random artist to the prompt.",
    "\u2199\ufe0f": "如果用户界面关键词语句为空.,则从关键词语或上一次中读取生成参数到用户界面/Read generation parameters from prompt or last generation if prompt is empty into user interface.",
    "\u{1f4c2}": "打开图像输出目录/Open images output directory",

    "修复图像的一部分/Inpaint a part of image": "在图像上画一个蒙版,脚本会根据关键词在蒙版遮罩的区域重新生成内容/Draw a mask over an image, and the script will regenerate the masked area with content according to prompt",
    "SD模式放大/SD upscale": "正常放大图像, 将结果分割为图块,使用img2img对每个图像进行改进,合并整个图像后返回/Upscale image normally, split result into tiles, improve each tile using img2img, merge whole image back",

    "只调整大小/Just resize": "调整图像的分辨率到目标分辨率。除非高度和宽度匹配,否则你将会得到不正常的横纵比/Resize image to target resolution. Unless height and width match, you will get incorrect aspect ratio.",
    "裁剪和调整大小/Crop and resize": "调整图像的分辨率大小,使得图像填充整个目标分辨率。会裁剪掉突出的部分/Resize the image so that entirety of target resolution is filled with the image. Crop parts that stick out.",
    "填充和调整大小/Resize and fill": "调整图像的分辨率大小,使整个图像在目标分辨率内/Resize the image so that entirety of image is inside target resolution. Fill empty space with image's colors.",

    "蒙版模糊程度/Mask blur": "处理蒙版前模糊多少,以像素为单位/How much to blur the mask before processing, in pixels.",
    "蒙版内容/Masked content": "在使用Stable Diffusion模型进行处理之前,要在蒙版区域放入什么/What to put inside the masked area before processing it with Stable Diffusion.",
    "填充/fill": "用图像的颜色填充它/fill it with colors of the image",
    "原始图像/original": "保留住原来的东西/keep whatever was there originally",
    "潜在噪声/latent noise": "用隐空间噪音填满它/fill it with latent space noise",
    "无潜在噪声/latent nothing": "用零点隐空间填满它/fill it with latent space zeroes",
    "全分辨率修复/Inpaint at full resolution": "将蒙版区域放大到目标分辨率进行修复,然后缩小尺寸并粘贴回原始图像中/Upscale masked region to target resolution, do inpainting, downscale back and paste into original image",

    "去噪强度/Denoising strength": "决定算法对于图像内容的遵守程度。为0时,什么都不会改变。为1时,你讲会得到一个毫不相关的图像。使用低于1的值时,处理所需要的步骤将少于指定的采样步骤/Determines how little respect the algorithm should have for image's content. At 0, nothing will change, and at 1 you'll get an unrelated image. With values below 1.0, processing will take less steps than the Sampling Steps slider specifies.",
    "去噪强度变化系数/Denoising strength change factor": "在图像迭代模式下,在每一次循环中,去噪强度乘以该值。小于1表示减少多样性,所以你的序列将收敛于固定的图像。大于1表示增加多样性,所以你的序列会变得越来越混乱/In loopback mode, on each loop the denoising strength is multiplied by this value. <1 means decreasing variety so your sequence will converge on a fixed picture. >1 means increasing variety so your sequence will become more and more chaotic.",

    "跳过/Skip": "停止生成当前的图像并继续生成后继图像/Stop processing current image and continue processing.",
    "终止/Interrupt": "停止处理图像并返回已生成的任意结果/Stop processing images and return any results accumulated so far.",
    "保存/Save": "将图像写到目录(默认为-log/images)并将生成的参数写入CSV文件/Write image to a directory (default - log/images) and generation parameters into csv file.",

    "X值/X values": "用逗号分隔X轴的值/Separate values for X axis using commas.",
    "Y值/Y values": "用逗号分隔Y轴的值/Separate values for Y axis using commas.",

    "None": "Do not do anything special",
    "关键词语句矩阵/Prompt matrix": "使用竖线字符(|)将关键词分隔为多个部分,脚本会为每一个组合创建一张图像(除了第一个部分,它会应用到所有组合里)/Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",
    "X/Y图/X/Y plot": "创建一个网格,其中的图像将会具有不同的参数。使用下面的输入来指定哪些参数将由列与行共享/Create a grid where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",
    "自定义python代码/Custom code": "运行python代码,仅限于高级用户。必须使用--allow-code运行程序才能正常工作/Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    "关键词S/R/Prompt S/R": "用逗号分隔单词列表,第一个单词将作为关键字:脚本会在关键词中搜索关键字,并替换为其他单词/Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",
    "关键词顺序/Prompt order": "用逗号分隔关键词列表,脚本将根据这些关键词的每一个合理顺序对关键词进行更改/Separate a list of words with commas, and the script will make a variation of prompt with those words for their every possible order",

    "无缝拼接/Tiling": "生成可以平铺的图像/Produce an image that can be tiled.",
    "贴图重叠范围/Tile overlap": "来自SD高级版, 贴图中间应该有多少像素重叠。因为是切片重叠,所以当它们合并回一张图像时,没有清晰可见的接缝/For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",
    
    "种子变异/Variation seed": "将不同的图像的图像生成种子混合到这一代/Seed of a different picture to be mixed into the generation.",
    "变异强度/Variation strength": "产生变异的程度。在0时,没有影响。在1时,你会得到变异种子的完成图像(除了遗传采样器,你只会得到一些东西)/How strong of a variation to produce. At 0, there will be no effect. At 1, you will get the complete picture with variation seed (except for ancestral samplers, where you will just get something).",
    "从高度上调整种子大小/Resize seed from height": "在指定分辨率下,尝试生成与相同种子生成图像相似的图像/Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",
    "从宽度上调整种子大小/Resize seed from width": "在指定分辨率下,尝试生成与相同种子生成图像相似的图像/Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",

    "图像查询/Interrogate": "从现有的图像重构关键词,并放入关键词语句中/Reconstruct prompt from existing image and put it into the prompt field.",

    "图像命名方式/Images filename pattern": "使用以下标签来定义如何选择图像的文件名:[steps], [cfg], [prompt],  [prompt_no_styles], [prompt_spaces], [width], [height], [styles], [sampler], [seed], [model_hash], [prompt_words], [date], [datetime], [job_timestamp];默认为空/Use following tags to define how filenames for images are chosen: [steps], [cfg], [prompt], [prompt_no_styles], [prompt_spaces], [width], [height], [styles], [sampler], [seed], [model_hash], [prompt_words], [date], [datetime], [job_timestamp]; leave empty for default.",
    "目录命名方式/Directory name pattern": "使用以下标签来定义如何选择图像和网格的子目录:: [steps], [cfg], [prompt], [prompt_no_styles], [prompt_spaces], [width], [height], [styles], [sampler], [seed], [model_hash], [prompt_words], [date], [datetime], [job_timestamp];默认为空/Use following tags to define how subdirectories for images and grids are chosen: [steps], [cfg], [prompt],  [prompt_no_styles], [prompt_spaces], [width], [height], [styles], [sampler], [seed], [model_hash], [prompt_words], [date], [datetime], [job_timestamp]; leave empty for default.",
    "最大关键词数量/Max prompt words": "在[关键词数量/prompt_words]选项中设置要使用的最大关键词数量;请注意:如果关键词语句太长,可能会超过系统可以处理文件路径的最大长度/Set the maximum number of words to be used in the [prompt_words] option; ATTENTION: If the words are too long, they may exceed the maximum length of the file path that the system can handle",
    
    "图像迭代/Loopback": "处理图像,将其作为输入对象,然后重复/Process an image, use it as an input, repeat.",
    "图像迭代次数/Loops": "需要重复处理一张图像,并将其作为下一个迭代的输入对象多少次/How many times to repeat processing an image and using it as input for the next iteration",

    "预设1/Style 1": "预设应用; 预设可以作用于正面关键词和负面关键词两种,并适用于两者/Style to apply; styles have components for both positive and negative prompts and apply to both",
    "预设2/Style 2": "预设应用; 预设可以作用于正面关键词和负面关键词两种,并适用于两者/Style to apply; styles have components for both positive and negative prompts and apply to both",
    "应用预设/Apply style": "将选择的预设插入到提示字段中/Insert selected styles into prompt fields",
    "创建预设/Create style": "将当前关键词提取到预设中,如果将标记关键词语句添加到文本中,关键词将在将来使用该关键词时,将其用作关键词语句的占位符/Save current prompts as a style. If you add the token {prompt} to the text, the style use that as placeholder for your prompt when you use the style in the future.",

    "模型名称/Checkpoint name": "在生成图像之前从模型加载权重。您可以使用hash值或部分文件名(见设置)作为模型名称。建议配合Y轴使用减少切换/Loads weights from checkpoint before making images. You can either use hash or a part of filename (as seen in settings) for checkpoint name. Recommended to use with Y axis for less switching.",

    "显存/vram": "Torch激活:在生成期间使用的VRAM(显存)的峰值量,不包括缓存的数据。\nTorch保留:Torch分配的VRAM的峰值量,包括所有活动和缓存的数据。\nSys VRAM:所有应用程序的VRAM分配峰值量/总数GPU VRAM(峰值利用率%)/Torch active: Peak amount of VRAM used by Torch during generation, excluding cached data.\nTorch reserved: Peak amount of VRAM allocated by Torch, including all active and cached data.\nSys VRAM: Peak amount of VRAM allocation across all applications / total GPU VRAM (peak utilization%).",

    "高分辨率图像修复/Highres. fix": "仅使用2步来创建一个小分辨率且高解析度的图像,然后再不改变构图的情况下改进图像细节/Use a two step process to partially create an image at smaller resolution, upscale, and then improve details in it without changing composition",
    "隐空间强度/Scale latent": "在隐空间中对图像进行缩放。另一种方法是从潜在的表象中产生完整的图像,将其升级,然后将其移回隐空间/Uscale the image in latent space. Alternative is to produce the full image from latent representation, upscale that, and then move it back to latent space.",

    "Eta噪声种子/Eta noise seed delta": "如果值不为0,它会被添加到'图像生成种子/seed'中,并且再使用带有Eta的采样工具时初始化噪声的RNG。你可以用它生成更多不一样的图像,或者你可以使用它来匹配其他软件的图像。/If this values is non-zero, it will be added to seed and used to initialize RNG for noises when using samplers with Eta. You can use this to produce even more variation of images, or you can use this to match images of other software if you know what you are doing.",
    "不在图像中添加水印/Do not add watermark to images": "如果启用此选项,将不会把水印添加到创建的图像中。警告:如果不添加水印.,你的行为可能是不道德的/If this option is enabled, watermark will not be added to created images. Warning: if you do not add watermark, you may be being in an unethical manner.",

    "正则文件名/Filename word regex": "该正则表达式将从文件名中提取关键词.,并使用下面的选项将它们连接到用于训练的标签文本中。留空以保持文件名文本原样/This regular expression will be used extract words from filename, and they will be joined using the option below into label text used for training. Leave empty to keep filename text as it is.",
    "文件名连接符号/Filename join string": "如果启用了上面的选项.,则此字符将用于将拆分的单词连接成一行/This string will be used to join split words into a single line if the option above is enabled.",

    "快速设置列表/Quicksettings list": "设置列表名称.,以逗号分隔.,用于设置应该进入顶部的快速访问栏.,而不是通常的设置选项。参见modules/shared.py设置名称。需要重新启动才能应用/List of setting names, separated by commas, for settings that should go to the quick access bar at the top, rather than the usual setting tab. See modules/shared.py for setting names. Requires restarting to apply.",

    "加权和/Weighted Sum": "结果=A * (1 - M) + B * M/Result = A * (1 - M) + B * M",
    "添加不同/Add difference": "结果=A + (B - C) * (1 - M)/Result = A + (B - C) * (1 - M)",
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
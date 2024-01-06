[English Version](README.md)

<p align="center"><img src="https://count.getloli.com/get/@sd-webui-bilingual-localization.github" alt="sd-webui-bilingual-localization"></p>

# sd-webui-bilingual-localization
[Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 双语对照翻译插件

![Snipaste_2023-03-30_01-05-45](https://user-images.githubusercontent.com/16256221/228617304-3107244b-ce13-4b96-b665-1d13090d24a7.png)

## 功能
- 全新实现的双语对照翻译功能，不必再担心切换翻译后找不到原始功能
- 兼容原生语言包扩展，无需重新导入多语言语料
- 支持动态title提示的翻译
- 额外支持作用域和正则表达式替换，翻译更加灵活

## 安装

以下方式选择其一，需要使用支持扩展功能的 webui <sup>(2023年之后的版本)</sup>

#### 方式1

使用 webui 提供的`Install from URL`功能安装

按下图所示，依次点击<kbd>Extensions</kbd> - <kbd>Install from URL</kbd>

然后在第一个文本框内填入`https://github.com/journey-ad/sd-webui-bilingual-localization`，点击<kbd>Install</kbd>按钮
![Snipaste_2023-02-28_00-27-48](https://user-images.githubusercontent.com/16256221/221625310-a6ef0b4c-a1e0-46bb-be9c-6d88cd0ad684.png)

之后切换到<kbd>Installed</kbd>面板，点击<kbd>Apply and restart UI</kbd>按钮
![Snipaste_2023-02-28_00-29-14](https://user-images.githubusercontent.com/16256221/221625345-9e656f25-89dd-4361-8ee5-f4ab39d18ca4.png)


#### 方式2

手动克隆到你的扩展目录里

```bash
git clone https://github.com/journey-ad/sd-webui-bilingual-localization extensions/sd-webui-bilingual-localization
```

## 使用

> **⚠️重要⚠️**   
> 确保<kbd>Settings</kbd> - <kbd>User interface</kbd> - <kbd>Localization</kbd> 已设置为了 `None`

在<kbd>Settings</kbd> - <kbd>Bilingual Localization</kbd>中选择要启用的本地化文件，依次点击<kbd>Apply settings</kbd>和<kbd>Reload UI</kbd>按钮
![Snipaste_2023-02-28_00-04-21](https://user-images.githubusercontent.com/16256221/221625729-73519629-8c1f-4eb5-99db-a1d3f4b58a87.png)

## 作用域支持

本地化语料支持限定作用域，防止影响全局翻译，语法规则：
- `##<SCOPE ID>##<TEXT>` 仅当节点祖先元素ID匹配指定的作用域时才会生效
- `##@<SELECTOR>##<TEXT>` 仅当节点祖先元素匹配指定的CSS选择器时才会生效

```json
{
  ...
  "##tab_ti##Normal": "正态", // 仅id="tab_ti"元素下的`Normal`会被翻译为`正态`
  "##tab_threedopenpose##Normal": "法线图", // 仅id="tab_threedopenpose"元素下的`Normal`会被翻译为`法线图`
  "##@.extra-networks .tab-nav button##Lora": "Lora模型", // 仅class=".extra-networks .tab-nav button"元素下的`Lora`会被翻译为`Lora模型`
  ...
}
```

## 正则表达式支持

本地化语料支持正则表达式替换，语法规则`@@<REGEXP>`，括号匹配变量`$n`，参考[String.prototype.replace()](https://developer.mozilla.org/docs/Web/JavaScript/Reference/Global_Objects/String/replace)
```json
{
  ...
  "@@/^(\\d+) images in this directory, divided into (\\d+) pages$/": "目录中有$1张图片，共$2页",
  "@@/^Favorites path from settings: (.*)$/": "设置的收藏夹目录：$1",
  ...
}
```

## 获取本地化文件

本地化文件不再随插件提供，请安装第三方语言包并按照本文[使用](#使用)部分的方式设置使用

*预览图片中的语言包可以在这里找到 https://gist.github.com/journey-ad/d98ed173321658be6e51f752d6e6163c*

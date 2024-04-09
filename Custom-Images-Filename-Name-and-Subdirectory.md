> the following information is about the image filename and subdirectory name, not the `Paths for saving \ Output directorys`
### By default, the Wub UI save images in the output directorys with a filename structure of

`number`-`seed`-`[prompt_spaces]`

```
01234-987654321-((masterpiece)), ((best quality)), ((illustration)), extremely detailed,style girl.png
```

A different image filename and optional subdirectory can be used if a user wishes.

Image filename pattern can be configured under.

`settings tab` > `Saving images/grids` > `Images filename pattern`

Subdirectory can be configured under settings.

`settings tab` > `Saving to a directory` > `Directory name pattern`

# Patterns
Web-Ui provides several patterns that can be used as placeholders for inserting information into the filename or subdirectory,
user can chain these patterns togetherm forming a filename that suits their use case.

| Pattern                        | Description                                          | Example                                                                                                                               |
|--------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `[seed]`                       | Seed                                                 | 1234567890                                                                                                                            |
| `[steps]`                      | Steps                                                | 20                                                                                                                                    |
| `[cfg]`                        | CFG scale                                            | 7                                                                                                                                     |
| `[sampler]`                    | Sampling method                                      | Euler a                                                                                                                               |
| `[model_name]`                 | name of the model                                    | sd-v1-4
| `[model_hash]`                 | Hash of the model                                    | 7460a6fa                                                                                                                              |
| `[width]`                      | Image width                                          | 512                                                                                                                                   |
| `[height]`                     | Image hight                                          | 512                                                                                                                                   |
| `[styles]`                     | Name of the chosen Styles                            | my style name                                                                                                                         |
| `[date]`                       | Date of the computer in ISO format                   | 2022-10-24                                                                                                                            |
| `[datetime]`                   | Datetime in "%Y%m%d%H%M%S"                           | 20221025013106                                                                                                                        |
| `[datetime<Format>]`           | Datetime in specified \<Format\>                       | \[datetime<%Y%m%d_%H%M%S_%f>]<br>20221025_014350_733877                                                                                   |
| `[datetime<Format><TimeZone>]` | Datetime at specific \<Time Zone\> in specified \<Format\> | \[datetime<%Y%m%d_%H%M%S_%f><Asia/Tokyo>]`<br>20221025_014350_733877                                                                                       |
| `[prompt_no_styles]`           | Prompt without Styles                                | 1gir,   white space, ((very   important)), [not important], (some value_1.5), (whatever), the end<br>                                     |
| `[prompt_spaces]`              | Prompt with Styles                                   | 1gir,   white space, ((very   important)), [not important], (some value_1.5), (whatever), the end<br>,   (((crystals texture Hair)))，((( |
| `[prompt]`                     | Prompt with Styles, `Space bar` replaced with`_`       | 1gir,\_\_\_white_space,\_((very\_important)),\_[not\_important],\_(some\_value\_1.5),\_(whatever),\_the\_end,\_(((crystals_texture_Hair)))，(((     |
| `[prompt_words]`               | Prompt   with Styles, Bracket and Comma removed      | 1gir white space very important not important some value 1 5 whatever the   end crystals texture Hair ， extremely detailed           |

### Datetime Formatting details
Reference python documentation for more details on [Format Codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)

### Datetime Time Zone details
Reference [List of Time Zones](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/List-of-Time-Zones) for a list of valid time zones

If `<Format>` is blank or invalid, it will use the default time format "%Y%m%d%H%M%S"
tip: you can use extra characters inside `<Format>` for punctuation, such as `_ -`

If `<TimeZone>` is blank or invalid, it will use the default system time zone

The Prompts and Style used for the above `[prompt]` examples

Prompt:
```
1girl,   white space, ((very important)), [not important], (some value:1.5), (whatever), the end
```
Selected Styles:
```
(((crystals texture Hair)))，(((((extremely detailed CG))))),((8k_wallpaper))
```

note: the `Styles` mentioned above is referring to the two drop down menu below the generate button

### if the Prompts is too long, it will be short
this is due to your computer having a maximum file length

# Add / Remove number to filename when saving
you can remove the prefix number 
by unchecking the checkbox under

`Setting` > `Saving images/grids` > `Add number to filename when saving`

with prefix number
```
00123-`987654321-((masterpiece)).png
```

without prefix number
```
987654321-((masterpiece)).png
```

### Caution
The purpose of the prefix number is to ensure that the saved image file name is **Unique**.
If you decide to not use the prefix number, make sure that your pattern will generate a unique file name,

**Otherwise files might be Overwritten**.

Generally datetime down to seconds should be able to guarantee that file name is unique.

```
[datetime<%Y%m%d_%H%M%S>]-[seed]
``` 
```
20221025_014350-281391998.png
```

But some **Custom Scripts** might generate **multiples images** using the **same seed** in a **single batch**,

in this case it is safer to also use `%f` for `Microsecond as a decimal number, zero-padded to 6 digits.`

```
[datetime<%Y%m%d_%H%M%S_%f>]-[seed]
```
```
20221025_014350_733877-281391998.png
```

# Filename Pattern Examples

If you're running Web-Ui on multiple machines, say on Google Colab and your own Computer, you might want to use a filename with a time as the Prefix.
this is so that when you download the files, you can put them in the same folder.

Also since you don't know what time zone Google Colab is using, you would want to specify the time zone.
```
[datetime<%Y%m%d_%H%M%S_%f><Asia/Tokyo>]-[seed]-[prompt_words]
```
```
20221025_032649_058536-3822510847-1girl.png
```

It might also be useful to set Subdirectory the date, so that one folder doesn't have too many images
```
[datetime<%Y-%m-%d><Asia/Tokyo>]
```
```
2022-10-25
```

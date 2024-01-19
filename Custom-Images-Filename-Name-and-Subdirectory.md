> the following information is about the image filename and subdirectory name, not the `Paths for saving \ Output directories`
### By default when Images filename pattern is blank, the Web UI saves images in the output directories and output archive with a filename structure of

Images: `[number]-[seed]` or `[number]-[seed]-[prompt_spaces]`

> the `[number]-` prefix is automatically added when [Add number to filename when saving](#add--remove-number-to-filename-when-saving) is enabled (default), it itself is not a [Pattern](#patterns).

```
01234-987654321-((masterpiece)), ((best quality)), ((illustration)), extremely detailed,style girl.png
```

Zip archive: `[datetime]_[[model_name]]_[seed]-[seed_last]`
```
20230530133149_[v1-5-pruned-emaonly]_987654321-987654329.zip
```

A different image filename and optional subdirectory and zip filename can be used if a user wishes.

Image filename pattern can be configured under.

`settings tab` > `Saving images/grids` > `Images filename pattern`

Subdirectory can be configured under settings.

`settings tab` > `Saving to a directory` > `Directory name pattern`

Zip archive can be configured under settings.

`settings tab` > `Saving images/grids` > `Archive filename pattern`

# Patterns
Web-Ui provides several patterns that can be used as placeholders for inserting information into the filename or subdirectory,
user can chain these patterns together, forming a filename that suits their use case.

| Pattern                        | Description                                          | Example                                                                                                                               |
|--------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `[seed]`                       | Seed                                                 | 1234567890                                                                                                                            |
| `[seed_first]`                 | First Seed of batch or Seed of single image          | [1234567890,1234567891,1234567892,1234567893] -> 1234567890<br>[1234567891] -> 1234567891
| `[seed_last]`                  | Last Seed of batch                                   | [1234567890,1234567891,1234567892,1234567893] -> 1234567893
| `[steps]`                      | Steps                                                | 20                                                                                                                                    |
| `[cfg]`                        | CFG scale                                            | 7                                                                                                                                     |
| `[sampler]`                    | Sampling method                                      | Euler a                                                                                                                               |
| `[model_name]`                 | Name of the model                                    | sd-v1-4
| `[model_hash]`                 | The first 8 characters of the prompt's SHA-256 hash  | 7460a6fa                                                                                                                              |
| `[width]`                      | Image width                                          | 512                                                                                                                                   |
| `[height]`                     | Image height                                          | 512                                                                                                                                   |
| `[styles]`                     | Name of the chosen Styles                            | my style name                                                                                                                         |
| `[date]`                       | Date of the computer in ISO format                   | 2022-10-24                                                                                                                            |
| `[datetime]`                   | Datetime in "%Y%m%d%H%M%S"                           | 20221025013106                                                                                                                        |
| `[datetime<Format>]`           | Datetime in specified \<Format\>                       | \[datetime<%Y%m%d_%H%M%S_%f>]<br>20221025_014350_733877                                                                                   |
| `[datetime<Format><TimeZone>]` | Datetime at specific \<Time Zone\> in specified \<Format\> | \[datetime<%Y%m%d_%H%M%S_%f><Asia/Tokyo>]`<br>20221025_014350_733877                                                                                       |
| `[job_timestamp]`  | job start time in "%Y%m%d%H%M%S" | 20221025013106 |
| `[prompt_no_styles]`           | Prompt without Styles                                | 1girl,   white space, ((very important)), [not important], (some value_1.5), (whatever), the end<br>                                     |
| `[prompt_spaces]`              | Prompt with Styles                                   | 1girl,   white space, ((very important)), [not important], (some value_1.5), (whatever), the end<br>,   (((crystals texture Hair)))，((( |
| `[prompt]`                     | Prompt with Styles, `Space bar` replaced with`_`       | 1girl,\_\_\_white_space,\_((very\_important)),\_[not\_important],\_(some\_value\_1.5),\_(whatever),\_the\_end,\_(((crystals_texture_Hair)))，(((     |
| `[prompt_words]`               | Prompt   with Styles, Bracket and Comma removed      | 1gir white space very important not important some value 1 5 whatever the   end crystals texture Hair ， extremely detailed           |
| `[prompt_hash]`<br>`[prompt_hash<N>]`  | The first 8  or `N` characters of the prompt's SHA-256 hash | 1girl -> 6362d0d2<br>(1girl:1.1) -> 0102e068 |
| `[negative_prompt_hash]`<br>`[negative_prompt_hash<N>]` | The first 8 or `N` characters of the negative prompt's SHA-256 hash | 1girl -> 6362d0d2<br>(1girl:1.1) -> 0102e068 |
| `[full_prompt_hash]`<br>`[full_prompt_hash<N>]` | The first 8 or `N` characters of the `<prompt> <negative_prompt>`'s SHA-256 hash | 1girl -> 6362d0d2<br>(1girl:1.1) -> 0102e068 |
| `[clip_skip]` | CLIP stop at last layers | 1 |
| `denoising` | denoising_strength if applicable | 0.5 |
| `[batch_number]` | the Nth image in a single batch job | BatchNo_[batch_number] -> BatchNo_3
| `[batch_size]`   | Batch size | [1234567890,1234567891,1234567892,1234567893] -> 4
| `[generation_number]` | the Nth image in an entire job | GenNo_[generation_number] -> GenNo_9
| `[hasprompt<prompt1\|default><prompt2>...]` | if specified `prompt` is found in prompts then `prompt` will be added to filename, else `default` will be added to filename (`default` can be blank) | [hasprompt<girl><boy>] -> girl<br>[hasprompt<girl\|no girl><boy\|no boy>] -> girlno boy
| `[user]` | the username used to login to webui when using `--gradio-auth username:pass` | username |
| `[image_hash]`<br>`[image_hash<N>]` | The first `N` characters or the full SHA-256 hash of the image (the image itself not the file) | 484a1e7a07e7573a9081ab6a527990bb4d410dc3 |
| `[none]` | Overrides the default, so you can get just the sequence number  |  |

If `<Format>` is blank or invalid, it will use the default time format "%Y%m%d%H%M%S"
tip: you can use extra characters inside `<Format>` for punctuation, such as `_ -`

If `<TimeZone>` is blank or invalid, it will use the default system time zone

If `batch size` is 1 the `[batch_number]`, `[seed_last]` along with the previous segment of text will not be added to filename

If `batch size` x `batch count` is 1 the [generation_number] along with the previous segment of text will not be added to filename

`[batch_number]` and `[generation_number]` along with the previous segment of text will not be added to filename of zip achive.

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
### Datetime Formatting details
Reference python documentation for more details on [Format Codes](https://docs.python.org/3.10/library/datetime.html#strftime-and-strptime-format-codes)

### Datetime Time Zone details
Reference [List of Time Zones](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/List-of-Time-Zones) for a list of valid time zones

### if the prompt is too long, it will be cutoff
this is due to your computer having a maximum file length

# Add / Remove number to filename when saving
you can remove the prefix number 
by unchecking the checkbox under

`Settings` > `Saving images/grids` > `Add number to filename when saving`

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
If you decide to not use the prefix number, make sure that your pattern will generate a unique file name, **otherwise files may be overwritten**.

Generally, datetime down to seconds should be able to guarantee that file name is unique.

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

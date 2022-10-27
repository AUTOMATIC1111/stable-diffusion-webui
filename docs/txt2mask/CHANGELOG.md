# Changelog
All notable changes to this project will be documented in this file.

## 0.1.1 - 23 September 2022
### Changed
- Download URL for weights replaced as the Github mirror was prone to hitting the download limit

## 0.1.0 - 22 September 2022
### Added
- New "Brush mask mode" option that allows you to specify what to do with the normal mask, i.e. the mask created using the brush tool or uploaded image
- Added debug mode

## 0.0.7 - 21 September 2022
### Changed
- Fixed issues related to negative mask prompt with images that are not 512x512

## 0.0.6 - 21 September 2022
### Added
- New "Negative mask prompt" field that allows you to subtract areas from your selection

### Changed
- Increased mask padding cap from 128 to 500, the padding is added to the image's original dimensions which means you might need a ton of padding if you're working with a large image

## 0.0.5 - 20 September 2022
### Added
- New "Mask Padding" option that allows you to easily expand the boundaries of the selected mask region

## 0.0.4 - 19 September 2022
### Added
- Support for mask prompt delimiter, use `|` to specify multiple selctions which will be stacked additively to produce the final mask image

### Changed
- Fixed a possible issue with the clipseg weight downloader
- Fix for crash related to inpainting at full resolution

## 0.0.3 - 19 September 2022
### Added
- Added option to show mask in output

## 0.0.2 - 18 September 2022
### Changed
- Readme and Changelog moved to `docs/txt2mask` so as not to conflict with Automatic's repo

### Removed
- Model weights are no longer included with the repo as they will not download properly through Github's website, instead the model files will be fetched by the script if they do not exist on your filesystem

## 0.0.1 - 17 September 2022
### Added
- Initial release
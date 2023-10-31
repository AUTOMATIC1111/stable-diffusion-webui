import gradio as gr
from modules import shared

shared.options_templates.update(shared.options_section(('canvas_hotkey', "Canvas Hotkeys"), {
    "canvas_hotkey_zoom": shared.OptionInfo("Alt", "Zoom canvas", gr.Radio, {"choices": ["Shift","Ctrl", "Alt"]}).info("If you choose 'Shift' you cannot scroll horizontally, 'Alt' can cause a little trouble in firefox"),
    "canvas_hotkey_adjust": shared.OptionInfo("Ctrl", "Adjust brush size", gr.Radio, {"choices": ["Shift","Ctrl", "Alt"]}).info("If you choose 'Shift' you cannot scroll horizontally, 'Alt' can cause a little trouble in firefox"),
    "canvas_hotkey_move": shared.OptionInfo("F", "Moving the canvas").info("To work correctly in firefox, turn off 'Automatically search the page text when typing' in the browser settings"),
    "canvas_hotkey_fullscreen": shared.OptionInfo("S", "Fullscreen Mode, maximizes the picture so that it fits into the screen and stretches it to its full width "),
    "canvas_hotkey_reset": shared.OptionInfo("R", "Reset zoom and canvas positon"),
    "canvas_hotkey_overlap": shared.OptionInfo("O", "Toggle overlap").info("Technical button, neededs for testing"),
    "canvas_show_tooltip": shared.OptionInfo(True, "Enable tooltip on the canvas"),
    "canvas_auto_expand": shared.OptionInfo(True, "Automatically expands an image that does not fit completely in the canvas area, similar to manually pressing the S and R buttons"),
    "canvas_blur_prompt": shared.OptionInfo(False, "Take the focus off the prompt when working with a canvas"),
    "canvas_disabled_functions": shared.OptionInfo(["Overlap"], "Disable function that you don't use", gr.CheckboxGroup, {"choices": ["Zoom","Adjust brush size", "Moving canvas","Fullscreen","Reset Zoom","Overlap"]}),
}))

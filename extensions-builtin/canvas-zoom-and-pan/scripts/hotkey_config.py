from modules import shared

shared.options_templates.update(shared.options_section(('canvas_hotkey', "Canvas Hotkeys"), {
    "canvas_hotkey_move": shared.OptionInfo("F", "Moving the canvas"),
    "canvas_hotkey_fullscreen": shared.OptionInfo("S", "Fullscreen Mode, maximizes the picture so that it fits into the screen and stretches it to its full width "),
    "canvas_hotkey_reset": shared.OptionInfo("R", "Reset zoom and canvas positon"),
    "canvas_hotkey_overlap": shared.OptionInfo("O", "Toggle overlap ( Technical button, neededs for testing )"),
    "canvas_show_tooltip": shared.OptionInfo(True, "Enable tooltip on the canvas"),
    "canvas_swap_controls": shared.OptionInfo(False, "Swap hotkey combinations for Zoom and Adjust brush resize"),
}))

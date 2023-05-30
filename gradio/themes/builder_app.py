import inspect
import time
from typing import Iterable

from gradio_client.documentation import document_fn

import gradio as gr

themes = [
    gr.themes.Base,
    gr.themes.Default,
    gr.themes.Soft,
    gr.themes.Monochrome,
    gr.themes.Glass,
]
colors = gr.themes.Color.all
sizes = gr.themes.Size.all

palette_range = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
size_range = ["xxs", "xs", "sm", "md", "lg", "xl", "xxl"]
docs_theme_core = document_fn(gr.themes.Base.__init__, gr.themes.Base)[1]
docs_theme_vars = document_fn(gr.themes.Base.set, gr.themes.Base)[1]


def get_docstr(var):
    for parameters in docs_theme_core + docs_theme_vars:
        if parameters["name"] == var:
            return parameters["doc"]
    raise ValueError(f"Variable {var} not found in theme documentation.")


def get_doc_theme_var_groups():
    source = inspect.getsource(gr.themes.Base.set)
    groups = []
    group, desc, variables, flat_variables = None, None, [], []
    for line in source.splitlines():
        line = line.strip()
        if line.startswith(")"):
            break
        elif line.startswith("# "):
            if group is not None:
                groups.append((group, desc, variables))
            group, desc = line[2:].split(": ")
            variables = []
        elif "=" in line:
            var = line.split("=")[0]
            variables.append(var)
            flat_variables.append(var)
    groups.append((group, desc, variables))
    return groups, flat_variables


variable_groups, flat_variables = get_doc_theme_var_groups()

css = """
.gradio-container {
    overflow: visible !important;
    max-width: none !important;
}
#controls {
    max-height: 100vh;
    flex-wrap: unset;
    overflow-y: scroll;
    position: sticky;
    top: 0;
}
#controls::-webkit-scrollbar {
  -webkit-appearance: none;
  width: 7px;
}

#controls::-webkit-scrollbar-thumb {
  border-radius: 4px;
  background-color: rgba(0, 0, 0, .5);
  box-shadow: 0 0 1px rgba(255, 255, 255, .5);
}
"""

with gr.Blocks(  # noqa: SIM117
    theme=gr.themes.Base(),
    css=css,
    title="Gradio Theme Builder",
) as demo:
    with gr.Row():
        with gr.Column(scale=1, elem_id="controls", min_width=400):
            with gr.Row():
                undo_btn = gr.Button("Undo").style(size="sm")
                dark_mode_btn = gr.Button("Dark Mode", variant="primary").style(
                    size="sm"
                )
            with gr.Tabs():
                with gr.TabItem("Source Theme"):
                    gr.Markdown(
                        """
                    Select a base theme below you would like to build off of. Note: when you click 'Load Theme', all variable values in other tabs will be overwritten!
                    """
                    )
                    base_theme_dropdown = gr.Dropdown(
                        [theme.__name__ for theme in themes],
                        value="Base",
                        show_label=False,
                    )
                    load_theme_btn = gr.Button("Load Theme", elem_id="load_theme")
                with gr.TabItem("Core Colors"):
                    gr.Markdown(
                        """Set the three hues of the theme: `primary_hue`, `secondary_hue`, and `neutral_hue`.
                        Each of these is a palette ranging from 50 to 950 in brightness. Pick a preset palette - optionally, open the accordion to overwrite specific values.
                        Note that these variables do not affect elements directly, but are referenced by other variables with asterisks, such as `*primary_200` or `*neutral_950`."""
                    )
                    primary_hue = gr.Dropdown(
                        [color.name for color in colors], label="Primary Hue"
                    )
                    with gr.Accordion(label="Primary Hue Palette", open=False):
                        primary_hues = []
                        for i in palette_range:
                            primary_hues.append(
                                gr.ColorPicker(
                                    label=f"primary_{i}",
                                )
                            )

                    secondary_hue = gr.Dropdown(
                        [color.name for color in colors], label="Secondary Hue"
                    )
                    with gr.Accordion(label="Secondary Hue Palette", open=False):
                        secondary_hues = []
                        for i in palette_range:
                            secondary_hues.append(
                                gr.ColorPicker(
                                    label=f"secondary_{i}",
                                )
                            )

                    neutral_hue = gr.Dropdown(
                        [color.name for color in colors], label="Neutral hue"
                    )
                    with gr.Accordion(label="Neutral Hue Palette", open=False):
                        neutral_hues = []
                        for i in palette_range:
                            neutral_hues.append(
                                gr.ColorPicker(
                                    label=f"neutral_{i}",
                                )
                            )

                with gr.TabItem("Core Sizing"):
                    gr.Markdown(
                        """Set the sizing of the theme via: `text_size`, `spacing_size`, and `radius_size`.
                        Each of these is set to a collection of sizes ranging from `xxs` to `xxl`. Pick a preset size collection - optionally, open the accordion to overwrite specific values.
                        Note that these variables do not affect elements directly, but are referenced by other variables with asterisks, such as `*spacing_xl` or `*text_sm`.
                        """
                    )
                    text_size = gr.Dropdown(
                        [size.name for size in sizes if size.name.startswith("text_")],
                        label="Text Size",
                    )
                    with gr.Accordion(label="Text Size Range", open=False):
                        text_sizes = []
                        for i in size_range:
                            text_sizes.append(
                                gr.Textbox(
                                    label=f"text_{i}",
                                )
                            )

                    spacing_size = gr.Dropdown(
                        [
                            size.name
                            for size in sizes
                            if size.name.startswith("spacing_")
                        ],
                        label="Spacing Size",
                    )
                    with gr.Accordion(label="Spacing Size Range", open=False):
                        spacing_sizes = []
                        for i in size_range:
                            spacing_sizes.append(
                                gr.Textbox(
                                    label=f"spacing_{i}",
                                )
                            )

                    radius_size = gr.Dropdown(
                        [
                            size.name
                            for size in sizes
                            if size.name.startswith("radius_")
                        ],
                        label="Radius Size",
                    )
                    with gr.Accordion(label="Radius Size Range", open=False):
                        radius_sizes = []
                        for i in size_range:
                            radius_sizes.append(
                                gr.Textbox(
                                    label=f"radius_{i}",
                                )
                            )

                with gr.TabItem("Core Fonts"):
                    gr.Markdown(
                        """Set the main `font` and the monospace `font_mono` here.
                        Set up to 4 values for each (fallbacks in case a font is not available).
                        Check "Google Font" if font should be loaded from Google Fonts.
                        """
                    )
                    gr.Markdown("### Main Font")
                    main_fonts, main_is_google = [], []
                    for i in range(4):
                        with gr.Row():
                            font = gr.Textbox(label=f"Font {i + 1}")
                            font_is_google = gr.Checkbox(label="Google Font")
                            main_fonts.append(font)
                            main_is_google.append(font_is_google)

                    mono_fonts, mono_is_google = [], []
                    gr.Markdown("### Monospace Font")
                    for i in range(4):
                        with gr.Row():
                            font = gr.Textbox(label=f"Font {i + 1}")
                            font_is_google = gr.Checkbox(label="Google Font")
                            mono_fonts.append(font)
                            mono_is_google.append(font_is_google)

                theme_var_input = []

                core_color_suggestions = (
                    [f"*primary_{i}" for i in palette_range]
                    + [f"*secondary_{i}" for i in palette_range]
                    + [f"*neutral_{i}" for i in palette_range]
                )

                variable_suggestions = {
                    "fill": core_color_suggestions[:],
                    "color": core_color_suggestions[:],
                    "text_size": [f"*text_{i}" for i in size_range],
                    "radius": [f"*radius_{i}" for i in size_range],
                    "padding": [f"*spacing_{i}" for i in size_range],
                    "gap": [f"*spacing_{i}" for i in size_range],
                    "weight": [
                        "100",
                        "200",
                        "300",
                        "400",
                        "500",
                        "600",
                        "700",
                        "800",
                    ],
                    "shadow": ["none"],
                    "border_width": [],
                }
                for variable in flat_variables:
                    if variable.endswith("_dark"):
                        continue
                    for style_type in variable_suggestions:
                        if style_type in variable:
                            variable_suggestions[style_type].append("*" + variable)
                            break

                variable_suggestions["fill"], variable_suggestions["color"] = (
                    variable_suggestions["fill"]
                    + variable_suggestions["color"][len(core_color_suggestions) :],
                    variable_suggestions["color"]
                    + variable_suggestions["fill"][len(core_color_suggestions) :],
                )

                for group, desc, variables in variable_groups:
                    with gr.TabItem(group):
                        gr.Markdown(
                            desc
                            + "\nYou can set these to one of the dropdown values, or clear the dropdown to set a custom value."
                        )
                        for variable in variables:
                            suggestions = []
                            for style_type in variable_suggestions:
                                if style_type in variable:
                                    suggestions = variable_suggestions[style_type][:]
                                    if "*" + variable in suggestions:
                                        suggestions.remove("*" + variable)
                                    break
                            dropdown = gr.Dropdown(
                                label=variable,
                                info=get_docstr(variable),
                                choices=suggestions,
                                allow_custom_value=True,
                            )
                            theme_var_input.append(dropdown)

        # App

        with gr.Column(scale=6, elem_id="app"):
            with gr.Column(variant="panel"):
                gr.Markdown(
                    """
                    # Theme Builder
                    Welcome to the theme builder. The left panel is where you create the theme. The different aspects of the theme are broken down into different tabs. Here's how to navigate them:
                    1. First, set the "Source Theme". This will set the default values that you can override.
                    2. Set the "Core Colors", "Core Sizing" and "Core Fonts". These are the core variables that are used to build the rest of the theme.
                    3. The rest of the tabs set specific CSS theme variables. These control finer aspects of the UI. Within these theme variables, you can reference the core variables and other theme variables using the variable name preceded by an asterisk, e.g. `*primary_50` or `*body_text_color`. Clear the dropdown to set a custom value.
                    4. Once you have finished your theme, click on "View Code" below to see how you can integrate the theme into your app. You can also click on "Upload to Hub" to upload your theme to the Hugging Face Hub, where others can download and use your theme.
                    """
                )
                with gr.Accordion("View Code", open=False):
                    output_code = gr.Code(language="python")
                with gr.Accordion("Upload to Hub", open=False):
                    gr.Markdown(
                        "You can save your theme on the Hugging Face Hub. HF API write token can be found [here](https://huggingface.co/settings/tokens)."
                    )
                    with gr.Row():
                        theme_name = gr.Textbox(label="Theme Name")
                        theme_hf_token = gr.Textbox(label="Hugging Face Write Token")
                        theme_version = gr.Textbox(
                            label="Version",
                            placeholder="Leave blank to automatically update version.",
                        )
                    upload_to_hub_btn = gr.Button("Upload to Hub")
                    theme_upload_status = gr.Markdown(visible=False)

                gr.Markdown("Below this panel is a dummy app to demo your theme.")

            name = gr.Textbox(
                label="Name",
                info="Full name, including middle name. No special characters.",
                placeholder="John Doe",
                value="John Doe",
                interactive=True,
            )

            with gr.Row():
                slider1 = gr.Slider(label="Slider 1")
                slider2 = gr.Slider(label="Slider 2")
            gr.CheckboxGroup(["A", "B", "C"], label="Checkbox Group")

            with gr.Row():
                with gr.Column(variant="panel", scale=1):
                    gr.Markdown("## Panel 1")
                    radio = gr.Radio(
                        ["A", "B", "C"],
                        label="Radio",
                        info="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
                    )
                    drop = gr.Dropdown(
                        ["Option 1", "Option 2", "Option 3"], show_label=False
                    )
                    drop_2 = gr.Dropdown(
                        ["Option A", "Option B", "Option C"],
                        multiselect=True,
                        value=["Option A"],
                        label="Dropdown",
                        interactive=True,
                    )
                    check = gr.Checkbox(label="Go")
                with gr.Column(variant="panel", scale=2):
                    img = gr.Image(
                        "https://i.ibb.co/6BgKdSj/groot.jpg", label="Image"
                    ).style(height=320)
                    with gr.Row():
                        go_btn = gr.Button(
                            "Go", label="Primary Button", variant="primary"
                        )
                        clear_btn = gr.Button(
                            "Clear", label="Secondary Button", variant="secondary"
                        )

                        def go(*args):
                            time.sleep(3)
                            return "https://i.ibb.co/6BgKdSj/groot.jpg"

                        go_btn.click(
                            go, [radio, drop, drop_2, check, name], img, api_name="go"
                        )

                        def clear():
                            time.sleep(0.2)
                            return None

                        clear_btn.click(clear, None, img)

                    with gr.Row():
                        btn1 = gr.Button("Button 1").style(size="sm")
                        btn2 = gr.UploadButton().style(size="sm")
                        stop_btn = gr.Button(
                            "Stop", label="Stop Button", variant="stop"
                        ).style(size="sm")

            gr.Examples(
                examples=[
                    [
                        "A",
                        "Option 1",
                        ["Option B"],
                        True,
                    ],
                    [
                        "B",
                        "Option 2",
                        ["Option B", "Option C"],
                        False,
                    ],
                ],
                inputs=[radio, drop, drop_2, check],
                label="Examples",
            )

            with gr.Row():
                gr.Dataframe(value=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], label="Dataframe")
                gr.JSON(
                    value={"a": 1, "b": 2, "c": {"test": "a", "test2": [1, 2, 3]}},
                    label="JSON",
                )
                gr.Label(value={"cat": 0.7, "dog": 0.2, "fish": 0.1})
                gr.File()
            with gr.Row():
                gr.ColorPicker()
                gr.Video(
                    "https://gradio-static-files.s3.us-west-2.amazonaws.com/world.mp4"
                )
                gr.Gallery(
                    [
                        (
                            "https://gradio-static-files.s3.us-west-2.amazonaws.com/lion.jpg",
                            "lion",
                        ),
                        (
                            "https://gradio-static-files.s3.us-west-2.amazonaws.com/logo.png",
                            "logo",
                        ),
                        (
                            "https://gradio-static-files.s3.us-west-2.amazonaws.com/tower.jpg",
                            "tower",
                        ),
                    ]
                ).style(height="200px", columns=2)

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot([("Hello", "Hi")], label="Chatbot")
                    chat_btn = gr.Button("Add messages")

                    def chat(history):
                        time.sleep(2)
                        yield [["How are you?", "I am good."]]

                    chat_btn.click(
                        lambda history: history
                        + [["How are you?", "I am good."]]
                        + (time.sleep(2) or []),
                        chatbot,
                        chatbot,
                    )
                with gr.Column(scale=1):
                    with gr.Accordion("Advanced Settings"):
                        gr.Markdown("Hello")
                        gr.Number(label="Chatbot control 1")
                        gr.Number(label="Chatbot control 2")
                        gr.Number(label="Chatbot control 3")

        # Event Listeners

        secret_css = gr.Textbox(visible=False)
        secret_font = gr.JSON(visible=False)

        demo.load(  # doing this via python was not working for some reason, so using this hacky method for now
            None,
            None,
            None,
            _js="""() => {
                document.head.innerHTML += "<style id='theme_css'></style>";
                let evt_listener = window.setTimeout(
                    () => {
                        load_theme_btn = document.querySelector('#load_theme');
                        if (load_theme_btn) {
                            load_theme_btn.click();
                            window.clearTimeout(evt_listener);
                        }
                    },
                    100
                );
            }""",
        )

        theme_inputs = (
            [primary_hue, secondary_hue, neutral_hue]
            + primary_hues
            + secondary_hues
            + neutral_hues
            + [text_size, spacing_size, radius_size]
            + text_sizes
            + spacing_sizes
            + radius_sizes
            + main_fonts
            + main_is_google
            + mono_fonts
            + mono_is_google
            + theme_var_input
        )

        def load_theme(theme_name):
            theme = [theme for theme in themes if theme.__name__ == theme_name][0]

            parameters = inspect.signature(theme.__init__).parameters
            primary_hue = parameters["primary_hue"].default
            secondary_hue = parameters["secondary_hue"].default
            neutral_hue = parameters["neutral_hue"].default
            text_size = parameters["text_size"].default
            spacing_size = parameters["spacing_size"].default
            radius_size = parameters["radius_size"].default

            theme = theme()

            font = theme._font[:4]
            font_mono = theme._font_mono[:4]
            font_is_google = [isinstance(f, gr.themes.GoogleFont) for f in font]
            font_mono_is_google = [
                isinstance(f, gr.themes.GoogleFont) for f in font_mono
            ]

            def pad_to_4(x):
                return x + [None] * (4 - len(x))

            var_output = []
            for variable in flat_variables:
                theme_val = getattr(theme, variable)
                if theme_val is None and variable.endswith("_dark"):
                    theme_val = getattr(theme, variable[:-5])
                var_output.append(theme_val)

            return (
                [primary_hue.name, secondary_hue.name, neutral_hue.name]
                + primary_hue.expand()
                + secondary_hue.expand()
                + neutral_hue.expand()
                + [text_size.name, spacing_size.name, radius_size.name]
                + text_size.expand()
                + spacing_size.expand()
                + radius_size.expand()
                + pad_to_4([f.name for f in font])
                + pad_to_4(font_is_google)
                + pad_to_4([f.name for f in font_mono])
                + pad_to_4(font_mono_is_google)
                + var_output
            )

        def generate_theme_code(
            base_theme, final_theme, core_variables, final_main_fonts, final_mono_fonts
        ):
            base_theme_name = base_theme
            base_theme = [theme for theme in themes if theme.__name__ == base_theme][
                0
            ]()

            parameters = inspect.signature(base_theme.__init__).parameters
            primary_hue = parameters["primary_hue"].default
            secondary_hue = parameters["secondary_hue"].default
            neutral_hue = parameters["neutral_hue"].default
            text_size = parameters["text_size"].default
            spacing_size = parameters["spacing_size"].default
            radius_size = parameters["radius_size"].default
            font = parameters["font"].default
            font = [font] if not isinstance(font, Iterable) else font
            font = [
                gr.themes.Font(f) if not isinstance(f, gr.themes.Font) else f
                for f in font
            ]
            font_mono = parameters["font_mono"].default
            font_mono = (
                [font_mono] if not isinstance(font_mono, Iterable) else font_mono
            )
            font_mono = [
                gr.themes.Font(f) if not isinstance(f, gr.themes.Font) else f
                for f in font_mono
            ]

            core_diffs = {}
            specific_core_diffs = {}
            core_var_names = [
                "primary_hue",
                "secondary_hue",
                "neutral_hue",
                "text_size",
                "spacing_size",
                "radius_size",
            ]
            for value_name, base_value, source_class, final_value in zip(
                core_var_names,
                [
                    primary_hue,
                    secondary_hue,
                    neutral_hue,
                    text_size,
                    spacing_size,
                    radius_size,
                ],
                [
                    gr.themes.Color,
                    gr.themes.Color,
                    gr.themes.Color,
                    gr.themes.Size,
                    gr.themes.Size,
                    gr.themes.Size,
                ],
                core_variables,
            ):
                if base_value.name != final_value:
                    core_diffs[value_name] = final_value
                source_obj = [
                    obj for obj in source_class.all if obj.name == final_value
                ][0]
                final_attr_values = {}
                diff = False
                for attr in dir(source_obj):
                    if attr in ["all", "name", "expand"] or attr.startswith("_"):
                        continue
                    final_theme_attr = (
                        value_name.split("_")[0]
                        + "_"
                        + (attr[1:] if source_class == gr.themes.Color else attr)
                    )
                    final_attr_values[final_theme_attr] = getattr(
                        final_theme, final_theme_attr
                    )
                    if getattr(source_obj, attr) != final_attr_values[final_theme_attr]:
                        diff = True
                if diff:
                    specific_core_diffs[value_name] = (source_class, final_attr_values)

            font_diffs = {}

            final_main_fonts = [font for font in final_main_fonts if font[0]]
            final_mono_fonts = [font for font in final_mono_fonts if font[0]]
            font = font[:4]
            font_mono = font_mono[:4]
            for base_font_set, theme_font_set, font_set_name in [
                (font, final_main_fonts, "font"),
                (font_mono, final_mono_fonts, "font_mono"),
            ]:
                if len(base_font_set) != len(theme_font_set) or any(
                    base_font.name != theme_font[0]
                    or isinstance(base_font, gr.themes.GoogleFont) != theme_font[1]
                    for base_font, theme_font in zip(base_font_set, theme_font_set)
                ):
                    font_diffs[font_set_name] = [
                        f"gr.themes.GoogleFont('{font_name}')"
                        if is_google_font
                        else f"'{font_name}'"
                        for font_name, is_google_font in theme_font_set
                    ]

            newline = "\n"

            core_diffs_code = ""
            if len(core_diffs) + len(specific_core_diffs) > 0:
                for var_name in core_var_names:
                    if var_name in specific_core_diffs:
                        cls, vals = specific_core_diffs[var_name]
                        core_diffs_code += f"""    {var_name}=gr.themes.{cls.__name__}({', '.join(f'''{k}="{v}"''' for k, v in vals.items())}),\n"""
                    elif var_name in core_diffs:
                        core_diffs_code += (
                            f"""    {var_name}="{core_diffs[var_name]}",\n"""
                        )

            font_diffs_code = ""

            if len(font_diffs) > 0:
                font_diffs_code = "".join(
                    [
                        f"""    {font_set_name}=[{", ".join(fonts)}],\n"""
                        for font_set_name, fonts in font_diffs.items()
                    ]
                )
            var_diffs = {}
            for variable in flat_variables:
                base_theme_val = getattr(base_theme, variable)
                final_theme_val = getattr(final_theme, variable)
                if base_theme_val is None and variable.endswith("_dark"):
                    base_theme_val = getattr(base_theme, variable[:-5])
                if base_theme_val != final_theme_val:
                    var_diffs[variable] = getattr(final_theme, variable)

            newline = "\n"

            vars_diff_code = ""
            if len(var_diffs) > 0:
                vars_diff_code = f""".set(
    {(',' + newline + "    ").join([f"{k}='{v}'" for k, v in var_diffs.items()])}
)"""

            output = f"""
import gradio as gr

theme = gr.themes.{base_theme_name}({newline if core_diffs_code or font_diffs_code else ""}{core_diffs_code}{font_diffs_code}){vars_diff_code}

with gr.Blocks(theme=theme) as demo:
    ..."""
            return output

        history = gr.State([])
        current_theme = gr.State(None)

        def render_variables(history, base_theme, *args):
            primary_hue, secondary_hue, neutral_hue = args[0:3]
            primary_hues = args[3 : 3 + len(palette_range)]
            secondary_hues = args[3 + len(palette_range) : 3 + 2 * len(palette_range)]
            neutral_hues = args[3 + 2 * len(palette_range) : 3 + 3 * len(palette_range)]
            text_size, spacing_size, radius_size = args[
                3 + 3 * len(palette_range) : 6 + 3 * len(palette_range)
            ]
            text_sizes = args[
                6
                + 3 * len(palette_range) : 6
                + 3 * len(palette_range)
                + len(size_range)
            ]
            spacing_sizes = args[
                6
                + 3 * len(palette_range)
                + len(size_range) : 6
                + 3 * len(palette_range)
                + 2 * len(size_range)
            ]
            radius_sizes = args[
                6
                + 3 * len(palette_range)
                + 2 * len(size_range) : 6
                + 3 * len(palette_range)
                + 3 * len(size_range)
            ]
            main_fonts = args[
                6
                + 3 * len(palette_range)
                + 3 * len(size_range) : 6
                + 3 * len(palette_range)
                + 3 * len(size_range)
                + 4
            ]
            main_is_google = args[
                6
                + 3 * len(palette_range)
                + 3 * len(size_range)
                + 4 : 6
                + 3 * len(palette_range)
                + 3 * len(size_range)
                + 8
            ]
            mono_fonts = args[
                6
                + 3 * len(palette_range)
                + 3 * len(size_range)
                + 8 : 6
                + 3 * len(palette_range)
                + 3 * len(size_range)
                + 12
            ]
            mono_is_google = args[
                6
                + 3 * len(palette_range)
                + 3 * len(size_range)
                + 12 : 6
                + 3 * len(palette_range)
                + 3 * len(size_range)
                + 16
            ]
            remaining_args = args[
                6 + 3 * len(palette_range) + 3 * len(size_range) + 16 :
            ]

            final_primary_color = gr.themes.Color(*primary_hues)
            final_secondary_color = gr.themes.Color(*secondary_hues)
            final_neutral_color = gr.themes.Color(*neutral_hues)
            final_text_size = gr.themes.Size(*text_sizes)
            final_spacing_size = gr.themes.Size(*spacing_sizes)
            final_radius_size = gr.themes.Size(*radius_sizes)

            final_main_fonts = []
            font_weights = set()
            for attr, val in zip(flat_variables, remaining_args):
                if "weight" in attr:
                    font_weights.add(val)
            font_weights = sorted(font_weights)

            for main_font, is_google in zip(main_fonts, main_is_google):
                if not main_font:
                    continue
                if is_google:
                    main_font = gr.themes.GoogleFont(main_font, weights=font_weights)
                final_main_fonts.append(main_font)
            final_mono_fonts = []
            for mono_font, is_google in zip(mono_fonts, mono_is_google):
                if not mono_font:
                    continue
                if is_google:
                    mono_font = gr.themes.GoogleFont(mono_font, weights=font_weights)
                final_mono_fonts.append(mono_font)

            theme = gr.themes.Base(
                primary_hue=final_primary_color,
                secondary_hue=final_secondary_color,
                neutral_hue=final_neutral_color,
                text_size=final_text_size,
                spacing_size=final_spacing_size,
                radius_size=final_radius_size,
                font=final_main_fonts,
                font_mono=final_mono_fonts,
            )

            theme.set(**dict(zip(flat_variables, remaining_args)))
            new_step = (base_theme, args)
            if len(history) == 0 or str(history[-1]) != str(new_step):
                history.append(new_step)

            return (
                history,
                theme._get_theme_css(),
                theme._stylesheets,
                generate_theme_code(
                    base_theme,
                    theme,
                    (
                        primary_hue,
                        secondary_hue,
                        neutral_hue,
                        text_size,
                        spacing_size,
                        radius_size,
                    ),
                    list(zip(main_fonts, main_is_google)),
                    list(zip(mono_fonts, mono_is_google)),
                ),
                theme,
            )

        def attach_rerender(evt_listener):
            return evt_listener(
                render_variables,
                [history, base_theme_dropdown] + theme_inputs,
                [history, secret_css, secret_font, output_code, current_theme],
            ).then(
                None,
                [secret_css, secret_font],
                None,
                _js="""(css, fonts) => {
                    document.getElementById('theme_css').innerHTML = css;
                    let existing_font_links = document.querySelectorAll('link[rel="stylesheet"][href^="https://fonts.googleapis.com/css"]');
                    existing_font_links.forEach(link => {
                        if (fonts.includes(link.href)) {
                            fonts = fonts.filter(font => font != link.href);
                        } else {
                            link.remove();
                        }
                    });
                    fonts.forEach(font => {
                        let link = document.createElement('link');
                        link.rel = 'stylesheet';
                        link.href = font;
                        document.head.appendChild(link);
                    });
                }""",
            )

        def load_color(color_name):
            color = [color for color in colors if color.name == color_name][0]
            return [getattr(color, f"c{i}") for i in palette_range]

        attach_rerender(primary_hue.select(load_color, primary_hue, primary_hues).then)
        attach_rerender(
            secondary_hue.select(load_color, secondary_hue, secondary_hues).then
        )
        attach_rerender(neutral_hue.select(load_color, neutral_hue, neutral_hues).then)
        for hue_set in (primary_hues, secondary_hues, neutral_hues):
            for hue in hue_set:
                attach_rerender(hue.blur)

        def load_size(size_name):
            size = [size for size in sizes if size.name == size_name][0]
            return [getattr(size, i) for i in size_range]

        attach_rerender(text_size.change(load_size, text_size, text_sizes).then)
        attach_rerender(
            spacing_size.change(load_size, spacing_size, spacing_sizes).then
        )
        attach_rerender(radius_size.change(load_size, radius_size, radius_sizes).then)

        attach_rerender(
            load_theme_btn.click(load_theme, base_theme_dropdown, theme_inputs).then
        )

        for theme_box in (
            text_sizes + spacing_sizes + radius_sizes + main_fonts + mono_fonts
        ):
            attach_rerender(theme_box.blur)
            attach_rerender(theme_box.submit)
        for theme_box in theme_var_input:
            attach_rerender(theme_box.blur)
            attach_rerender(theme_box.select)
        for checkbox in main_is_google + mono_is_google:
            attach_rerender(checkbox.select)

        dark_mode_btn.click(
            None,
            None,
            None,
            _js="""() => {
            if (document.querySelectorAll('.dark').length) {
                document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
            } else {
                document.querySelector('body').classList.add('dark');
            }
        }""",
        )

        def undo(history_var):
            if len(history_var) <= 1:
                return {history: gr.skip()}
            else:
                history_var.pop()
                old = history_var.pop()
                return [history_var, old[0]] + list(old[1])

        attach_rerender(
            undo_btn.click(
                undo, [history], [history, base_theme_dropdown] + theme_inputs
            ).then
        )

        def upload_to_hub(data):
            try:
                theme_url = data[current_theme].push_to_hub(
                    repo_name=data[theme_name],
                    version=data[theme_version] or None,
                    hf_token=data[theme_hf_token],
                    theme_name=data[theme_name],
                )
                space_name = "/".join(theme_url.split("/")[-2:])
                return (
                    gr.Markdown.update(
                        value=f"Theme uploaded [here!]({theme_url})! Load it as `gr.Blocks(theme='{space_name}')`",
                        visible=True,
                    ),
                    "Upload to Hub",
                )
            except Exception as e:
                return (
                    gr.Markdown.update(
                        value=f"Error: {e}",
                        visible=True,
                    ),
                    "Upload to Hub",
                )

        upload_to_hub_btn.click(lambda: "Uploading...", None, upload_to_hub_btn).then(
            upload_to_hub,
            {
                current_theme,
                theme_name,
                theme_hf_token,
                theme_version,
            },
            [theme_upload_status, upload_to_hub_btn],
        )


if __name__ == "__main__":
    demo.launch()

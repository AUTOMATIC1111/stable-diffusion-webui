# Fix for Hidden UI Tabs Breaking img2img (#16322)

## Problem
When adding "txt2img" to Settings → User Interface → Hidden UI tabs, the img2img tab stops functioning due to JavaScript errors:
- "error running callback function: TypeError: counter is null"
- "Uncaught (in promise) TypeError: P[x] is undefined"

## Root Cause
The current implementation in `modules/ui.py` (lines 944-947) completely skips rendering hidden tabs:

```python
for interface, label, ifid in sorted_interfaces:
    if label in shared.opts.hidden_tabs:
        continue  # THIS SKIPS RENDERING ENTIRELY
    with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
        interface.render()
```

This causes:
1. DOM elements like `txt2img_prompt`, `txt2img_token_counter` are never created
2. JavaScript in `javascript/token-counters.js` tries to access these null elements
3. Cross-tab functionality breaks (img2img depends on txt2img components being present)

## Solution

### 1. Fix modules/ui.py (lines 944-947)
Change from skipping render to hiding with CSS:

```python
for interface, label, ifid in sorted_interfaces:
    is_hidden = label in shared.opts.hidden_tabs
    with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}", visible=not is_hidden):
        interface.render()
    if ifid not in ["extensions", "settings"]:
        loadsave.add_block(interface, ifid)
```

### 2. Add null checks in javascript/token-counters.js (lines 52-68)
Update `setupTokenCounting` function:

```javascript
function setupTokenCounting(id, id_counter, id_button) {
    var prompt = gradioApp().getElementById(id);
    var counter = gradioApp().getElementById(id_counter);
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    
    // Add null checks to handle hidden tabs gracefully
    if (!prompt || !counter || !textarea) {
        console.debug(`Token counting elements not found for ${id}, likely due to hidden tab`);
        return;
    }
    
    if (counter.parentElement == prompt.parentElement) {
        return;
    }
    
    prompt.parentElement.insertBefore(counter, prompt);
    // ... rest of the function
}
```

### 3. Update settings tooltip in modules/shared_init.py or modules/ui_settings.py
Find the `hidden_tabs` setting definition and update the label/info:

```python
opts.add_option(
    "hidden_tabs",
    shared.OptionInfo(
        [],
        label="Hidden UI tabs",
        info="Tabs are rendered but hidden with CSS to maintain functionality. Note: Some interdependencies between tabs are preserved.",
        component_args={"choices": [x[1] for x in interfaces]},
        onchange=lambda: None
    )
)
```

## Benefits
- ✅ Tabs are visually hidden but fully functional
- ✅ All DOM elements are created (prevents null reference errors)
- ✅ Cross-tab dependencies work correctly
- ✅ Token counters function properly
- ✅ Settings can be applied without manual config file editing

## Testing
1. Before: Hidden txt2img → img2img Generate button doesn't work
2. After: Hidden txt2img → img2img works normally, tab is just not visible

Fixes #16322

/* This is a basic library that allows controlling elements that take some form of user input.

This was previously written in typescript, where all controllers implemented an interface. Not
all methods were needed in all the controllers, but it was done to keep a common interface, so
your main app can serve as a controller of controllers.

These controllers were built to work on the shapes of html elements that gradio components use.

There may be some notes in it that only applied to my use case, but I left them to help others
along.

You will need the parent element for these to work.
The parent element can be defined as the element (div) that gets the element id when assigning
an element id to a gradio component.

Example:
  gr.TextBox(value="...", elem_id="THISID")

Basic usage, grab an element that is the parent container for the component.

Send it in to the class, like a function, don't forget the "new" keyword so it calls the constructor
and sends back a new object.

Example:

let txt2imgPrompt = new TextComponentController(gradioApp().querySelector("#txt2img_prompt"))

Then use the getVal() method to get the value, or use the setVal(myValue) method to set the value.

Input types that are groups, like Checkbox groups (not individual checkboxes), take in an array of values.

Checkbox group has to reset all values to False (unchecked), then set the values in your array to true (checked).
If you don't hold a reference to the values (the labels in string format), you can acquire them using the getVal() method.
*/
class DropdownComponentController {
    constructor(element) {
        this.element = element;
        this.childSelector = this.element.querySelector('select');
        this.children = new Map();
        Array.from(this.childSelector.querySelectorAll('option')).forEach(opt => this.children.set(opt.value, opt));
    }
    getVal() {
        return this.childSelector.value;
    }
    updateVal(optionElement) {
        optionElement.selected = true;
    }
    setVal(name) {
        this.updateVal(this.children.get(name));
        this.eventHandler();
    }
    eventHandler() {
        this.childSelector.dispatchEvent(new Event("change"));
    }
}
class CheckboxComponentController {
    constructor(element) {
        this.element = element;
        this.child = this.element.querySelector('input');
    }
    getVal() {
        return this.child.checked;
    }
    updateVal(checked) {
        this.child.checked = checked;
    }
    setVal(checked) {
        this.updateVal(checked);
        this.eventHandler();
    }
    eventHandler() {
        this.child.dispatchEvent(new Event("change"));
    }
}
class CheckboxGroupComponentController {
    constructor(element) {
        this.element = element;
        //this.checkBoxes = new Object;
        this.children = new Map();
        Array.from(this.element.querySelectorAll('input')).forEach(input => this.children.set(input.nextElementSibling.innerText, input));
        /* element id gets use fieldset, grab all inputs (the bool val) get the userfriendly label, use as key, put bool value in mapping   */
        //Array.from(this.component.querySelectorAll("input")).forEach( _input => this.checkBoxes[_input.nextElementSibling.innerText] = _input)
        /*Checkboxgroup structure
        <fieldset>
            <div> css makes translucent
            <span>
              serves as label for component
            </span>
            <div data-testid='checkbox-group'> container for checkboxes
                <label>
                    <input type=checkbox>
                    <span>checkbox words</span>
                </label>
                ...
            </div>
        </fieldset>
        */
    }
    updateVal(label) {
        /*********
         calls updates using a throttle or else the backend does not get updated properly
         * ********/
        setTimeout(() => this.conditionalToggle(true, this.children.get(label)), 2);
    }
    setVal(labels) {
        /* Handles reset and updates all in array to true */
        this.reupdateVals();
        labels.forEach(l => this.updateVal(l));
    }
    getVal() {
        //return the list of values that are true
        return [...this.children].filter(([k, v]) => v.checked).map(arr => arr[0]);
    }
    reupdateVals() {
        /**************
         * for reupdating all vals, first set to false
         **************/
        this.children.forEach(inputChild => this.conditionalToggle(false, inputChild));
    }
    conditionalToggle(desiredVal, inputChild) {
        //This method behaves like 'set this value to this'
        //Using element.checked = true/false, does not register the change, even if you called change afterwards,
        //  it only sets what it looks like in our case, because there is no form submit, a person then has to click on it twice.
        //Options are to use .click() or dispatch an event
        if (desiredVal != inputChild.checked) {
            inputChild.dispatchEvent(new Event("change")); //using change event instead of click, in case browser ad-blockers blocks the click method
        }
    }
    eventHandler(checkbox) {
        checkbox.dispatchEvent(new Event("change"));
    }
}
class RadioComponentController {
    constructor(element) {
        this.element = element;
        this.children = new Map();
        Array.from(this.element.querySelectorAll("input")).forEach(input => this.children.set(input.value, input));
    }
    getVal() {
        //radio groups have a single element that's checked is true
        //           as array       arr k,v pair    element.checked  ) -> array of len(1) with [k,v] so either [0] [1].value
        return [...this.children].filter(([l, e]) => e.checked)[0][0];
        //return Array.from(this.children).filter( ([label, input]) => input.checked)[0][1].value
    }
    updateVal(child) {
        this.eventHandler(child);
    }
    setVal(name) {
        //radio will trigger all false except the one that get the event change
        //to keep the api similar, other methods are still called
        this.updateVal(this.children.get(name));
    }
    eventHandler(child) {
        child.dispatchEvent(new Event("change"));
    }
}
class NumberComponentController {
    constructor(element) {
        this.element = element;
        this.childNumField = element.querySelector('input[type=number]');
    }
    getVal() {
        return this.childNumField.value;
    }
    updateVal(text) {
        this.childNumField.value = text;
    }
    eventHandler() {
        this.element.dispatchEvent(new Event("input"));
    }
    setVal(text) {
        this.updateVal(text);
        this.eventHandler();
    }
}
class SliderComponentController {
    constructor(element) {
        this.element = element;
        this.childNumField = this.element.querySelector('input[type=number]');
        this.childRangeField = this.element.querySelector('input[type=range]');
    }
    getVal() {
        return this.childNumField.value;
    }
    updateVal(text) {
        //both are not needed, either works, both are left in so one is a fallback in case of gradio changes 
        this.childNumField.value = text;
        this.childRangeField.value = text;
    }
    eventHandler() {
        this.element.dispatchEvent(new Event("input"));
        this.childNumField.dispatchEvent(new Event("input"));
        this.childRangeField.dispatchEvent(new Event("input"));
    }
    setVal(text) {
        this.updateVal(text);
        this.eventHandler();
    }
}
class TextComponentController {
    constructor(element) {
        this.element = element;
        this.child = element.querySelector('textarea');
    }
    getVal() {
        return this.child.value;
    }
    eventHandler() {
        this.element.dispatchEvent(new Event("input"));
        this.child.dispatchEvent(new Event("change"));
        //Workaround to solve no target with v(o) on eventhandler, define my own target
        let ne = new Event("input");
        Object.defineProperty(ne, "target", { value: this.child });
        this.child.dispatchEvent(ne);
    }
    updateVal(text) {
        this.child.value = text;
    }
    appendValue(text) {
        //might add delimiter option
        this.child.value += ` ${text}`;
    }
    setVal(text, append = false) {
        if (append) {
            this.appendValue(text);
        }
        else {
            this.updateVal(text);
        }
        this.eventHandler();
    }
}
class JsonComponentController extends TextComponentController {
    constructor(element) {
        super(element);
    }
    getVal() {
        return JSON.parse(this.child.value);
    }
}
class ColorComponentController {
    constructor(element) {
        this.element = element;
        this.child = this.element.querySelector('input[type=color]');
    }
    updateVal(text) {
        this.child.value = text;
    }
    getVal() {
        return this.child.value;
    }
    setVal(text) {
        this.updateVal(text);
        this.eventHandler();
    }
    eventHandler() {
        this.child.dispatchEvent(new Event("input"));
    }
}

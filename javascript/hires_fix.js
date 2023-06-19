
function setInactive(elem, inactive){
    if(inactive){
        elem.classList.add('inactive')
    } else{
        elem.classList.remove('inactive')
    }
}

function onCalcResolutionHires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y){
    hrUpscaleBy = gradioApp().getElementById('txt2img_hr_scale')
    hrResizeX = gradioApp().getElementById('txt2img_hr_resize_x')
    hrResizeY = gradioApp().getElementById('txt2img_hr_resize_y')

    gradioApp().getElementById('txt2img_hires_fix_row2').style.display = opts.use_old_hires_fix_width_height ? "none" : ""

    setInactive(hrUpscaleBy, opts.use_old_hires_fix_width_height || hr_resize_x > 0 || hr_resize_y > 0)
    setInactive(hrResizeX, opts.use_old_hires_fix_width_height || hr_resize_x == 0)
    setInactive(hrResizeY, opts.use_old_hires_fix_width_height || hr_resize_y == 0)

    return [enable, width, height, hr_scale, hr_resize_x, hr_resize_y]
}

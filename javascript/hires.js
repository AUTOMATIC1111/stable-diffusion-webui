/* global gradioApp, opts */
function onCalcResolutionHires(enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y) {
  function setInactive(elem, inactive) {
    elem.classList.toggle('inactive', !!inactive);
  }
  const hrUpscaleBy = gradioApp().getElementById('txt2img_hr_scale');
  const hrResizeX = gradioApp().getElementById('txt2img_hr_resize_x');
  const hrResizeY = gradioApp().getElementById('txt2img_hr_resize_y');
  gradioApp().getElementById('txt2img_hires_fix_row3').style.display = opts.use_old_hires_fix_width_height ? 'none' : '';
  setInactive(hrUpscaleBy, opts.use_old_hires_fix_width_height || hr_resize_x > 0 || hr_resize_y > 0);
  setInactive(hrResizeX, opts.use_old_hires_fix_width_height || hr_resize_x == 0);
  setInactive(hrResizeY, opts.use_old_hires_fix_width_height || hr_resize_y == 0);
  return [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y];
}

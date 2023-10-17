function onCalcResolutionHires(width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler) {
  const setInactive = (elem, inactive) => elem.classList.toggle('inactive', !!inactive);
  const hrUpscaleBy = gradioApp().getElementById('txt2img_hr_scale');
  const hrResizeX = gradioApp().getElementById('txt2img_hr_resize_x');
  const hrResizeY = gradioApp().getElementById('txt2img_hr_resize_y');
  setInactive(hrUpscaleBy, hr_resize_x > 0 || hr_resize_y > 0);
  setInactive(hrResizeX, hr_resize_x === 0);
  setInactive(hrResizeY, hr_resize_y === 0);
  return [width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler];
}

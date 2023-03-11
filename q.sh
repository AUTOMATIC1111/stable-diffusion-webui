echo 'README.md
extensions-builtin/Lora/ui_extra_networks_lora.py
javascript/progressbar.js
modules/api/api.py
modules/api/models.py
modules/models/diffusion/uni_pc/sampler.py
modules/sd_hijack.py
modules/ui_extra_networks_checkpoints.py
modules/ui_extra_networks_hypernets.py
modules/ui_extra_networks_textual_inversion.py
test/basic_features/txt2img_test.py' | while read F; do
  echo "Creating backup" $F
  echo $F | cpio -pvd /tmp/backup
  echo "Copying original" $F
  echo ~/branches/automatic/$F | cpio -pvd .
done

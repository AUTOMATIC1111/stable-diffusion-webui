import os
import modules.launch_utils as lu


def install():
    lu.prepare_environment()
    # 安装gfpgan
    lu.run_pip("install --verbose  gfpgan==1.3.8", "install gfpgan")
    # 安装GroundingDINO ，先克隆源码再从源码安装
    lu.git_clone("https://github.com/IDEA-Research/GroundingDINO.git", lu.repo_dir("GroundingDINO"), "GroundingDINO")
    lu.run_pip(f"install  -e ${lu.repo_dir('GroundingDINO')}", "install GroundingDINO")


def download_config():
    import modules.paths_internal as pi
    # 下载obs文件
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/configs/Tags-zh-full-pack.csv -O ${pi.data_path}/extensions/tagcomplete/tags/tags-zh-full-pack.csv")
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/configs/config.json -O  ${pi.data_path}/config.json")
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/zh_cn.csv -O ${pi.data_path}/extensions/tagcomplete/tags/zh_cn.csv")
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/artists.txt -O ${pi.data_path}/stable-diffusion-webui/interrogate/artists.txt")
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/flavors.txt -O ${pi.data_path}/stable-diffusion-webui/interrogate/flavors.txt")
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/mediums.txt -O ${pi.data_path}/stable-diffusion-webui/interrogate/mediums.txt")
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/movements.txt -O ${pi.data_path}/stable-diffusion-webui/interrogate/movements.txt")
    lu.run(
        f"wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/libcudart.so -O /opt/conda/lib/libcudart.so")


def clone_extensions():
    import modules.paths_internal as pi
    lu.git_clone("https://github.com/CompVis/taming-transformers.git",
                 os.path.join(pi.extensions_dir, "taming-transformers"),
                 "clone taming-transformers")
    lu.git_clone("https://github.com/nonnonstop/sd-webui-3d-open-pose-editor",
                 os.path.join(pi.extensions_dir, "sd-webui-3d-open-pose-editor"),
                 "clone sd-webui-3d-open-pose-editor")
    lu.git_clone("https://github.com/KutsuyaYuki/ABG_extension",
                 os.path.join(pi.extensions_dir, "ABG_extension"),
                 "clone ABG_extension")
    lu.git_clone("https://github.com/Jackstrawcd/sd-webui-additional-networks.git",
                 os.path.join(pi.extensions_dir, "sd-webui-additional-networks.git"),
                 "clone sd-webui-additional-networks.git")
    lu.git_clone("https://github.com/Bing-su/adetailer.git",
                 os.path.join(pi.extensions_dir, "adetailer"),
                 "clone adetailer")
    lu.git_clone("https://github.com/deforum-art/sd-webui-deforum",
                 os.path.join(pi.extensions_dir, "deforum"),
                 "clone deforum")
    lu.git_clone("https://github.com/AlUlkesh/stable-diffusion-webui-images-browser",
                 os.path.join(pi.extensions_dir, "images-browser"),
                 "clone images-browser")
    lu.git_clone("https://github.com/hako-mikan/sd-webui-lora-block-weight",
                 os.path.join(pi.extensions_dir, "lora-block-weight"),
                 "clone lora-block-weight")
    lu.git_clone("https://github.com/hnmr293/posex",
                 os.path.join(pi.extensions_dir, "posex"),
                 "clone posex")
    lu.git_clone("https://jihulab.com/xiaolxl_pub/sd-webui-prompt-all-in-one",
                 os.path.join(pi.extensions_dir, "prompt-all-in-one"),
                 "clone prompt-all-in-one")
    lu.git_clone("https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg.git",
                 os.path.join(pi.extensions_dir, "rembg"),
                 "clone rembg")

    lu.git_clone("https://github.com/Jackstrawcd/sd-webui-controlnet.git",
                 os.path.join(pi.extensions_dir, "sd-webui-controlnet"),
                 "clone sd-webui-controlnet")
    lu.git_clone("https://github.com/jexom/sd-webui-depth-lib.git",
                 os.path.join(pi.extensions_dir, "sd-webui-depth-lib"),
                 "clone sd-webui-depth-lib")

    lu.git_clone("https://github.com/Jackstrawcd/sd-webui-llul.git",
                 os.path.join(pi.extensions_dir, "sd-webui-llul"),
                 "clone sd-webui-llul")
    lu.git_clone("https://github.com/continue-revolution/sd-webui-segment-anything.git",
                 os.path.join(pi.extensions_dir, "segment-anything"),
                 "clone segment-anything")
    lu.git_clone("https://github.com/a2569875/stable-diffusion-webui-composable-lora",
                 os.path.join(pi.extensions_dir, "stable-diffusion-webui-composable-lora"),
                 "clone stable-diffusion-webui-composable-lora")
    lu.git_clone("https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN",
                 os.path.join(pi.extensions_dir, "stable-diffusion-webui-localization-zh_CN"),
                 "clone stable-diffusion-webui-localization-zh_CN")
    lu.git_clone("https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen",
                 os.path.join(pi.extensions_dir, "stable-diffusion-webui-promptgen"),
                 "clone stable-diffusion-webui-promptgen")
    lu.git_clone("https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer.git",
                 os.path.join(pi.extensions_dir, "stable-diffusion-webui-tokenizer"),
                 "clone stable-diffusion-webui-tokenizer")
    lu.git_clone("https://github.com/picobyte/stable-diffusion-webui-wd14-tagger.git",
                 os.path.join(pi.extensions_dir, "stable-diffusion-webui-wd14-tagger"),
                 "clone stable-diffusion-webui-wd14-tagger")
    lu.git_clone("https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git",
                 os.path.join(pi.extensions_dir, "tagcomplete"),
                 "clone tagcomplete")
    lu.git_clone("https://github.com/opparco/stable-diffusion-webui-two-shot.git",
                 os.path.join(pi.extensions_dir, "two-shot"),
                 "clone two-shot")
    lu.git_clone("https://github.com/xilai0715/sd-vide-frame.git",
                 os.path.join(pi.extensions_dir, "video--frame"),
                 "clone video--frame")

    lu.git_clone("https://gitlab.ilongyuan.cn/qzai/sd_super_functions.git",
                 os.path.join(pi.extensions_dir, "sd_super_functions"),
                 "clone sd_super_functions")

    lu.git_clone("https://gitlab.ilongyuan.cn/aigc/sd-webui-filemanager.git",
                 os.path.join(pi.extensions_dir, "sd-webui-filemanager"),
                 "clone sd-webui-filemanager")


def install_worker_requirements():
    from handlers import get_task_handlers
    for h in get_task_handlers():
        h.install_requirements()


def main():
    install()
    download_config()



if __name__ == '__main__':
    main()

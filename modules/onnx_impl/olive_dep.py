from installer import log, installed, install


def install_olive():
    if installed("olive-ai"):
        log.debug("Olive: olive-ai is already installed. Skipping olive-ai installation.")
        return

    install("olive-ai", "olive-ai")
    log.info("Olive: olive-ai is installed. Please restart webui session.")

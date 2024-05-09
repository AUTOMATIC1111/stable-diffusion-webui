from modules import launch_utils

prepare_environment = launch_utils.prepare_environment


def main():
    launch_utils.startup_timer.record("Environment setup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        prepare_environment()


if __name__ == "__main__":
    main()

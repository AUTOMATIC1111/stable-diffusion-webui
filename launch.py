from modules import codemaster

args = codemaster.args
python = codemaster.python
git = codemaster.git
index_url = codemaster.index_url
dir_repos = codemaster.dir_repos

commit_hash = codemaster.commit_hash
git_tag = codemaster.git_tag

run = codemaster.run
is_installed = codemaster.is_installed
repo_dir = codemaster.repo_dir

run_pip = codemaster.run_pip
check_run_python = codemaster.check_run_python
git_clone = codemaster.git_clone
git_pull_recursive = codemaster.git_pull_recursive
list_extensions = codemaster.list_extensions
run_extension_installer = codemaster.run_extension_installer
prepare_environment = codemaster.prepare_environment
configure_for_tests = codemaster.configure_for_tests
start = codemaster.start


def main():
    if args.dump_sysinfo:
        filename = codemaster.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    codemaster.startup_timer.record("initial startup")

    with codemaster.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            prepare_environment()

    if args.test_server:
        configure_for_tests()

    start()


if __name__ == "__main__":
    main()

# NOTE:
# This file downloads the doxa_cli to allow you to upload an agent.
# You do NOT need to edit this file, use it as is.

import sys

if sys.version_info[0] != 3:
    print("Please run this script using python3")
    sys.exit(1)

import json
import os
import platform
import stat
import subprocess
import tarfile
import urllib.error
import urllib.request


# Returns `windows`, `darwin` (macos) or `linux`
def get_os():
    system = platform.system()

    if system == "Linux":
        # The exe version works better for WSL
        if "microsoft" in platform.platform():
            return "windows"

        return "linux"
    elif system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "darwin"
    else:
        raise Exception(f"Unknown platform {system}")


def get_bin_name():
    bin_name = "doxa_cli"
    if get_os() == "windows":
        bin_name = "doxa_cli.exe"
    return bin_name


def get_bin_dir():
    return os.path.join(os.path.dirname(__file__), "bin")


def get_binary():
    return os.path.join(get_bin_dir(), get_bin_name())


def install_binary():
    match_release = None
    try:
        match_release = sys.argv[1]
    # Arguments are not required
    except IndexError:
        pass

    REPO_RELEASE_URL = "https://api.github.com/repos/louisdewar/doxa/releases/latest"
    try:
        f = urllib.request.urlopen(REPO_RELEASE_URL)
    except urllib.error.URLError:
        print("There was an SSL cert verification error")
        print(
            'If you are on a mac and you have recently installed a new version of\
     python then you should navigate to "/Applications/Python {VERSION}/"'
        )
        print('Then run a script in that folder called "Install Certificates.command"')
        sys.exit(1)

    response = json.loads(f.read())

    print("Current version tag:", response["tag_name"])

    assets = [
        asset for asset in response["assets"] if asset["name"].endswith(".tar.gz")
    ]

    # Find the release for this OS
    match_release = get_os()
    try:
        asset_choice = next(asset for asset in assets if match_release in asset["name"])
        print(
            'Automatically picked {} to download based on match "{}"\n'.format(
                asset_choice["name"], match_release
            )
        )
    except StopIteration:
        print('Couldn\'t find "{}" in releases'.format(match_release))
        sys.exit(1)

    download_url = asset_choice["browser_download_url"]

    # Folder where this script is + bin
    bin_dir = get_bin_dir()

    print("Downloading", asset_choice["name"], "to", bin_dir)
    print("({})".format(download_url))

    # Clear bin directory if it exists
    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)

    zip_path = os.path.join(bin_dir, asset_choice["name"])

    # Download zip file
    urllib.request.urlretrieve(download_url, zip_path)

    # Open and extract zip file
    tar_file = tarfile.open(zip_path)
    tar_file.extractall(bin_dir)
    tar_file.close()

    # Delete zip file
    os.remove(zip_path)

    # Path to the actual binary program (called doxa_cli or doxa_cli.exe)
    bin_name = get_bin_name()
    binary_path = os.path.join(bin_dir, bin_name)

    if not os.path.exists(binary_path):
        print(f"Couldn't find the binary file `{bin_name}` in the bin directory")
        print("This probably means that there was a problem with the download")
        sys.exit(1)

    if get_os() != "windows":
        # Make binary executable
        st = os.stat(binary_path)
        os.chmod(binary_path, st.st_mode | stat.S_IEXEC)

    # Run help
    print("Installed binary\n\n")


def run_command(args):
    bin_path = get_binary()

    if not os.path.exists(bin_path):
        install_binary()
    subprocess.call([bin_path] + args)


if __name__ == "__main__":
    run_command(sys.argv[1:])

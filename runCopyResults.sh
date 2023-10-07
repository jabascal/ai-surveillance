#!/usr/bin/bash

# Set default value for USER_NAME
USER_NAME=${1:-$USER}

# Path to the directory where Docker saves the results
SUBFOLDER_NAME="$(date +%Y%m%d)"
PATH_VOLUME="/var/lib/docker/volumes/surveillance/_data"
echo -e "\n Files in volume $PATH_VOLUME :"
ls $PATH_VOLUME

# Path to the directory where the results will be copied
PATH_OUT="/home/$USER_NAME/Pictures/Surveillance"
echo -e "\n Path target: $PATH_OUT :"
if [ ! -d "$PATH_OUT" ]; then
    mkdir "$PATH_OUT"
    echo "Created directory $PATH_OUT"
fi

FOLDERS=$(ls $PATH_VOLUME)
echo "\n Folders in volume: $FOLDERS"

# List all folders in the source directory
for FOLDER in $FOLDERS; do
    PATH_FOLDER="$PATH_VOLUME/$FOLDER"

    # Check if folder name starts with todays date
    # Copy folder to target directory
    cp -r "$PATH_FOLDER" "$PATH_OUT"
    chmod -R 777 "$PATH_OUT/$FOLDER"
    echo "Copied $PATH_FOLDER"

    # Remove folder from source directory
    rm -r "$PATH_FOLDER"
done


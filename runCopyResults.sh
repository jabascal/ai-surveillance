#!/usr/bin/bash
# Run as: sudo runCopyResults.sh [USER_NAME] [COPY_FREQUENCY]

# Set default value for USER_NAME
#USER_NAME=${1:-$USER}
USER_NAME=${1:-abascal}

# Set default value for COPY_FREQUENCY if not passed as argument
COPY_FREQUENCY=${2:-5}

# Path to the directory where the results will be copied
PATH_OUT="/home/$USER_NAME/Pictures/Surveillance"
echo -e "\n Path target: $PATH_OUT :"
if [ ! -d "$PATH_OUT" ]; then
    mkdir "$PATH_OUT"
    echo "Created directory $PATH_OUT"
fi

# Path to the directory where Docker saves the results
SUBFOLDER_NAME="$(date +%Y%m%d)"
PATH_VOLUME="/var/lib/docker/volumes/surveillance/_data"
echo -e "\n Files in volume $PATH_VOLUME :"
ls $PATH_VOLUME

# Find all folders in the volume that start with today's date
FOLDERS=$(find "$PATH_VOLUME" -type d -name "$SUBFOLDER_NAME*")
echo -e "\n Folders to copy: $FOLDERS"

# Loop over each folder and copy its contents to the target directory
# if they were modified within the last COPY_FREQUENCY minutes
for FOLDER in $FOLDERS; do
    echo -e "\n Folder name: $FOLDER"

    # Create target directory if it does not exist
    FOLDER_NAME=$(basename "$FOLDER")    
    TARGET_DIR="$PATH_OUT/$FOLDER_NAME"
    if [ ! -d "$TARGET_DIR" ]; then
        mkdir -p "$TARGET_DIR"
        echo "Created directory $TARGET_DIR"
    fi

    # Find all files modified within the last COPY_FREQUENCY minutes
    FILES=$(find "$FOLDER" -type f -mmin -"$COPY_FREQUENCY")

    # Loop over each file and copy it to the target directory
    for FILE in $FILES; do
        cp "$FILE" "$TARGET_DIR"
        # Change permissions of the copied file
        chmod 777 "$TARGET_DIR/$(basename "$FILE")"
        echo "Copied $FILE"
    done
done

# Change permissions of the target directory and its contents
chmod 777 "$TARGET_DIR"


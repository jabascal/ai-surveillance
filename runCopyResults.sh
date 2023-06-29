#!/usr/bin/bash

# Path to the directory where Docker saves the results
SUBFOLDER_NAME="$(date +%Y%m%d)"
PATH_VOLUME="/var/lib/docker/volumes/surveillance/_data/"

# Path to the directory where the results will be copied
PATH_OUT="/home/$USER/Pictures/Surveillance"
if [ ! -d "$PATH_OUT" ]; then
    mkdir "$PATH_OUT"
fi

# List all folders in the source directory
for FOLDER in "$PATH_VOLUME"/*; do
    # Check if folder name starts with "20230629"
    if [[ "$FOLDER" == *"$SUBFOLDER_NAME"* ]]; then

        # Copy folder to target directory
        cp -r "$FOLDER" "$PATH_OUT"
        rm -r "$FOLDER"
    fi
done

# List recent files 
#FILES=$(find $PATH_VOLUME -mtime -$TIME_COPY) 
#for FILE in $FILES    
#    do       
#        cp "$FILE" $PATH_OUT"$(basename $FILE)" 
#done
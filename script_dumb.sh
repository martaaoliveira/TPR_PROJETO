#!/bin/bash

remote_user="labcom"
remote_host="10.10.10.1"
remote_path="~/Desktop/"
remote_password="labcom"

while true; do
    # Generate a random size for the file (from 1 KB to 100 MB)
    file_size_kb=$(( ( RANDOM % 100000 ) + 1 ))
    file_name="example${file_size_kb}KB.txt"

    # Create a random file with the specified size
    dd if=/dev/urandom of="$file_name" bs=1024 count="$file_size_kb" 2>/dev/null

    # Print start of file transfer
    echo "Transferring $file_name..."

    # Send the file via SCP using sshpass
    sshpass -p "$remote_password" scp -o StrictHostKeyChecking=no "$file_name" "$remote_user"@"$remote_host":"$remote_path"

    # Print end of file transfer
    echo "File $file_name transferred."

    # Delete the local file
    rm "$file_name"

    # Generate a random sleep time (from 1 to 30 seconds)
    sleep_time=$(( ( RANDOM % 30 ) + 1 ))
    sleep "$sleep_time"
done
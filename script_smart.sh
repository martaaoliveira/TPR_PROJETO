remote_user="labcom"
remote_host="10.10.10.1"
remote_path="~/Desktop/"
remote_password="labcom"
# Lista de comandos possíveis
commands=("ls" "mkdir" "nano" "echo" "pwd" "date" "cat" "grep" "touch" "whoami","lscpu","netstat","ip a","traceroute")

while true; do

        #gerar ficheiros de 100 KB e 100000 KB (0.1 mb e 100mb)
        file_size_kb=$(( ( RANDOM % 99501 ) + 100 ))

    # se o ficheiro for pequeno entre 50kb e 250 kb
    if [ "$file_size_kb" -ge 50 ] && [ "$file_size_kb" -le 250 ]; then
        file_name="example${file_size_kb}KB.txt"
        dd if=/dev/urandom of="$file_name" bs=1024 count="$file_size_kb" 2>/dev/null
        echo "Transferring $file_name..."
        sshpass -p "$remote_password" scp -o StrictHostKeyChecking=no "$file_name" "$remote_user"@"$remote_host":"$remote_path"
        echo "File $file_name transferred."
        rm "$file_name"
    fi
    # se for grande vai dividir em chunks 
    else
        chunk_size=$(( ( RANDOM % 200 ) + 50 )) # Random chunk size between 50KB and 250KB
        file_name="example${file_size_kb}KB.txt"
        dd if=/dev/urandom of="$file_name" bs=1024 count="$file_size_kb" 2>/dev/null
        split -b "${chunk_size}K" "$file_name" "$file_name-part" # Splitting the file into chunks
        rm "$file_name" # Remove the original file
        
        chunks=( $(ls "$file_name"-part*) ) # List of chunks
        
        # Randomly shuffle chunks
        for ((i = ${#chunks[@]} - 1; i > 0; i--)); do
            j=$((RANDOM % (i + 1)))
            temp="${chunks[i]}"
            chunks[i]="${chunks[j]}"
            chunks[j]="$temp"
        done
        
        for chunk in "${chunks[@]}"; do
            echo "Transferring $chunk..."
            sshpass -p "$remote_password" scp -o StrictHostKeyChecking=no "$chunk" "$remote_user"@"$remote_host":"$remote_path"
            echo "Chunk $chunk transferred."
            rm "$chunk"

            
            # Random gaussian delay between copies
            mean=85
            deviation=70
            delay=$(python -c "import random, math; print(int(random.gauss($mean, $deviation)))")
            sleep "$delay"
        done
    fi


 # Execute a random number of commands (4 to 5) with Gaussian delays(mean 2, desv 1 )
    num_commands=$(( ( RANDOM % 2 ) + 4 )) # Random number of commands
    for (( i=1; i<=num_commands; i++ )); do
        random_command=${commands[$RANDOM % ${#commands[@]} ]}
        echo "Executing command: $random_command"
        eval "$random_command"
        mean_command_delay=2
        deviation_command_delay=1
        command_delay=$(python -c "import random, math; print(int(random.gauss($mean_command_delay, $deviation_command_delay)))")
        sleep "$command_delay"
    done
    
    # Gaussian delay between attacks ( with a mean of 120 seconds and a deviation of 60 seconds)
    mean_sleep_delay=120
    deviation_sleep_delay=60
    attack_delay=$(python -c "import random, math; print(int(random.gauss($mean_sleep_delay, $deviation_sleep_delay)))")
    sleep "$attack_delay"
done

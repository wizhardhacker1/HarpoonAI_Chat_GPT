![HAI](https://github.com/user-attachments/assets/d07b1ae0-3d9e-46ca-8fab-c4c22a046771)

Save
Copy
1
2
3
4
5
6
7
8
9
10
11
12
13
⌄
⌄
⌄
⌄
# HarpoonAI - Offline AI Chat with Llama 7B and Falcon 40B (Uncensored)

This project provides an offline AI chat interface using uncensored versions of Llama 7B and Falcon 40B.

## Download & Installation

1. **Download:** 
2. **Unzip:** Extract the downloaded archive.
3. **Copy:** Move the `HarpoonAI` folder to your `/home` directory or `/opt`.
4. **Permissions:**
   ```bash
   cd HarpoonAI
   chmod +x *.sh
Run Installer:
bash
Copy
1
./installer.sh
Running HarpoonAI
bash
Save
Copy
1
2
cd HarpoonAI
./startAI.sh
Stopping and Restarting
If you stop the application using CTRL-C and need to restart, you may need to replace the index.html and server.py files with the original versions from the HarpoonAI folder. This step may be necessary due to potential file modifications during runtime.

Frontend Update:
bash
Copy
1
2
3
cd offline_ai_chat/frontend/
mv index.html index.old  # Backup existing file
cp ../../HarpoonAI/index.html . # Replace with the original
Backend Update:
bash
Copy
1
2
3
cd ../backend/
mv server.py server.old  # Backup existing file
cp ../../HarpoonAI/server.py . # Replace with the original
Restart:
bash
Copy
1
2
cd ../../HarpoonAI
./startAI.sh
Tested Environment
Fedora

Models Included
Llama 7B Uncensored
Falcon 40B Uncensored
Disclaimer
This project provides access to uncensored language models. Use responsibly and be aware of the potential for generating offensive or harmful content. The developers are not responsible for any misuse of this software.

Troubleshooting
Issue: Encountering errors after restarting.
Solution: Ensure you've followed the "Stopping and Restarting" instructions to replace the index.html and server.py files.
Issue: Installer fails.
Solution: Double-check dependencies and ensure you have the necessary permissions. Provide more details about the error message for further assistance.
Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.

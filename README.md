![HAI](https://github.com/user-attachments/assets/d07b1ae0-3d9e-46ca-8fab-c4c22a046771)


   # HarpoonAI - Offline AI Chat with Llama 7B and Falcon 40B (Uncensored)

This project provides an offline AI chat interface using uncensored versions of Llama 7B and Falcon 40B.

## Download & Installation

1. **Download:** [Download HarpoonAI](link-to-your-download)
2. **Unzip:** Extract the downloaded archive.
3. **Copy:** Move the `HarpoonAI` folder to your `/home` directory or `/opt`.
4. **Permissions:**
   ```bash
   cd HarpoonAI
   chmod +x *.sh

   Run Installer
   ./installer.sh

   Stopping and Restarting
If you stop the application using CTRL-C and need to restart, you may need to replace the index.html and server.py files with the original versions from the HarpoonAI folder. This step may be necessary due to potential file modifications during runtime.

Update Frontend index.html

cd offline_ai_chat/frontend/
mv index.html index.old  # Backup existing file
cp ../../HarpoonAI/index.html . # Replace with the original

Update Backend server.py

cd ../backend/
mv server.py server.old  # Backup existing file
cp ../../HarpoonAI/server.py . # Replace with the original

cd/HarpoonAI
./startAI.sh

open browser
localhost:8000 or 127.0.0.1:8000

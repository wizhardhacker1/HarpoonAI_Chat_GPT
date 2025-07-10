![HAI](https://github.com/user-attachments/assets/d07b1ae0-3d9e-46ca-8fab-c4c22a046771)

```markdown
# HarpoonAI - Offline AI Chat with Llama 7B and Falcon 40B (Uncensored)

This project provides an offline AI chat interface using uncensored versions of Llama 7B and Falcon 40B.

## Download & Installation

1. **Download:** https://github.com/wizhardhacker1/HarpoonAI/blob/main/HarpoonAI.zip
2. **Unzip:** Extract the downloaded archive.
3. **Copy:** Move the `HarpoonAI` folder to your `/home` directory or `/opt`.
4. **Permissions:**
   ```bash
   cd HarpoonAI
   chmod +x *.sh
   ```
5. **Run Installer:**
   ```bash
   ./installer.sh
   ```
   !! STOP IF RUNNING _ Replace the `index.html` and `server.py` files with the original versions from the `HarpoonAI` folder.  This step is necessary due to upgraded files !!


1. **Frontend Update:**
   ```bash
   cd offline_ai_chat/frontend/
   mv index.html index.old  # Backup existing file
   cp ../../HarpoonAI/index.html . # Replace with the original
   ```
2. **Backend Update:**
   ```bash
   cd ../backend/
   mv server.py server.old  # Backup existing file
   cp ../../HarpoonAI/server.py . # Replace with the original
   ```

## Running HarpoonAI

```bash
cd HarpoonAI
./startAI.sh
```


## Tested Environment

Fedora server/workstation  and RedHat Server

## Models Included

* Llama 7B Uncensored
* Falcon 40B Uncensored


## Disclaimer

This project provides access to uncensored language models.  Use responsibly and be aware of the potential for generating offensive or harmful content.  The developers are not responsible for any misuse of this software.


## Troubleshooting

* **Issue:**  Encountering errors after restarting.
* **Solution:**  Ensure you've followed the "Stopping and Restarting" instructions to replace the `index.html` and `server.py` files.

* **Issue:** Installer fails.
* **Solution:** Double-check dependencies and ensure you have the necessary permissions. Provide more details about the error message for further assistance.


## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.


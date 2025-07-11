<img width="905" height="883" alt="HarpoonAIv2" src="https://github.com/user-attachments/assets/1b841897-dc76-4f96-a750-0f0dc23a2376" />


```markdown
# HarpoonAI - Offline AI Chat with Llama 7B and Falcon 40B (Uncensored)

This project provides an offline AI chat interface using uncensored versions of Llama 7B and Falcon 40B.

## Download & Installation

1. **Download:** 
3. **Create:** `HarpoonAI` folder to your `/home` directory or `/opt`.
4. **Permissions:**
   ```bash
   cd HarpoonAI
   copy install.sh to `HarpoonAI` folder
   chmod +x *.sh
   ```
5. **Run Installer:**
   ```bash
   ./install.sh
   ```
🌐 Web Interface: http://localhost:8000
🤖 LLaMA Server: http://localhost:8080
🦅 Falcon Server: http://localhost:8081

✅ FIXES APPLIED:
   • URL ingestion endpoint fixed (POST method)
   • Document search algorithm improved with better scoring
   • Enhanced error handling and user feedback
   • Better file processing for multiple formats
   • Improved web content extraction with retry logic
   • Duplicate content detection

📁 Log Files:
   LLaMA:   /home/harper/offline_ai_chat/llama_server.log
   Falcon:  /home/harper/offline_ai_chat/falcon_server.log
   Backend: /home/harper/offline_ai_chat/backend_server.log

🎉 To start the FIXED system:
   /home/harper/offline_ai_chat/start_harpoonai.sh

🛑 To stop all services:
   /home/harper/offline_ai_chat/stop_harpoonai.sh

🌐 Once started, access: http://localhost:8000

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


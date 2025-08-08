<img width="1351" height="994" alt="Screenshot From 2025-07-11 20-46-31" src="https://github.com/user-attachments/assets/3e9ce229-c515-43f1-965c-f8c6c37ef379" />
<img width="314" height="253" alt="Screenshot_20250807_212829" src="https://github.com/user-attachments/assets/c30bbb8e-1491-4b9a-9934-125bee40e201" />


```markdown
# HarpoonAI - Offline AI Chat with Llama 7B and Falcon 40B (Uncensored) w/GPU support

This project provides an offline AI chat interface using uncensored versions of Llama 7B and Falcon 40B.

## Download & Installation

1. **Download:** 
3. **Create:** `HarpoonAI` folder to your `/home` directory or `/opt`.

   cd HarpoonAI
   copy install.sh to `HarpoonAI` folder
   chmod +x *.sh
   ```
4. **Run Installer:**

   ./install.sh

🌐 Web Interface: http://localhost:8000
🤖 LLaMA Server: http://localhost:8080
🦅 Falcon Server: http://localhost:8081

✅ Vector embeddings for semantic understanding

✅ ChromaDB vector database for intelligent search

✅ Improved document chunking with sentence-transformers

✅ Hybrid search combining semantic + keyword matching

✅ Enhanced conversation context management 
----------------------------------------------------------------
📁 Log Files:
   LLaMA:   /home/USER/offline_ai_chat/llama_server.log
   
   Falcon:  /home/USER/offline_ai_chat/falcon_server.log
   
   Backend: /home/USER/offline_ai_chat/backend_server.log

----------------------------------------------------------------

🎉 To start the FIXED system:

   /home/USER/offline_ai_chat/start_harpoonai.sh

---------------------------------------------------------------- 

🛑 To stop all services:

   /home/USER/offline_ai_chat/stop_harpoonai.sh

----------------------------------------------------------------

🌐 Once started, access: http://localhost:8000

----------------------------------------------------------------
## Tested On
* Fedora server
* Fedora workstation
* RedHat Server

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


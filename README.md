# HarpoonAI
HarpoonAI Private GPT 



Tested on Fedora 


copy HarpoonAI folder to /home directory or /opt

Run -  chmod +x *sh

run installer - 
	cd HarpoonAI
		./installer.sh

afterwards " if running CTRL-C to stop 

goto 
	cd offline_ai_chat 
		cd frontend/ 
( rename old prior)   mv index.html index.old
replace index.html with one in the HarpoonAI folder

then

cd backend/
( rename old prior)   mv server.py server.old
replace server.py with one in the HarpoonAI folder ( rename old prior)


After that cd back out to HarpoonAI and then run
./startAI.sh

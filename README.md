# Intel_AI_Hackathon
Repository for Intel Python HackFury2

# Team Name - ALPHA_AI

Team Members:
Nikhil Mattapally
Pruthvi Raj R G
Om Shri Prasath
Aditya Swaroop

Please refer to the ppt for detailed description

# For implementing the Sentiment Analysis Server:

1. Download the weights for the network from the link - https://drive.google.com/open?id=1TFGD3cWP9FV4TYgE60dKtdrpjoSEfOK6
into the server advanced folder.

2. Now run the following to install the dependencies: pip install -r requirements_1.txt

3. Run the following commands to inflate the server: 
export FLASK_APP=app_advanced.py
flask run -host=0.0.0.0 --port=5000

# For implementing the Sentiment Analysis Server:

1. Download the weights for the network from the link - https://drive.google.com/open?id=1OmrvjlXFvx2tLvDwB4f22ax_NgEOiQ-f
into the tmp folder inside server_simple_classifier folder.

2. Now run the following to install the dependencies: pip install -r requirements_2.txt

3. Run the following commands to inflate the server: 
export FLASK_APP=final_bert_app.py
flask run -host=0.0.0.0 --port=5050

Now both the servers will be up and running.
Download and install the apk in an android.

Later on opening the app you will be prompted to enter the ip-adress of the server.(The machine hosting and the android should be connected to the same wifi network)

The app will be completely functional for testing now.

# Please refer to the video to find out more on how to interact with the app.



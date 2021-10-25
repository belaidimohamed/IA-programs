d={}
import os
import subprocess
os.chdir(r"C:\Users\user16\Desktop")
import speech_recognition as sr
r=sr.Recognizer()
with sr.Microphone() as source :
    print("speek freely : ")
    audio=r.listen(source)
    try :
        text=r.recognize_google(audio)
        print(text)
        subprocess.Popen(text,shell=True)

    except :
        print('sorry')

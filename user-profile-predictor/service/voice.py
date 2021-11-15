#!/usr/bin/env python3

import speech_recognition as sr
from flask import Flask

# get audio from the microphone
r = sr.Recognizer()

app = Flask(__name__)


@app.route('/voice', methods=['GET'])
def speech():
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))


if __name__ == '__main__':
    app.run(debug=True)

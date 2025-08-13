# Send notifications to my phone
import requests as r
from import_env import *
def notif(title, message, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/449px-Tensorflow_logo.svg.png"):
    data={
        "token":PUSHOVER_TOKEN,
        "user":PUSHOVER_USER_KEY,
        "title": title,
        "message": message,
        "url": image_url,
        "url_title": "image"
    }
    r.post("https://api.pushover.net/1/messages.json", data=data)
if __name__ == "__main__":
    notif("Test Notification", "This a test notification that was sent from your personnal computer!", "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/449px-Tensorflow_logo.svg.png")
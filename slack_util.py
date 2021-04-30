import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

def send_message(message):

    try:
        response = client.chat_postMessage(
            channel="D01S9QSPVJN",
            text=message
        )

    except SlackApiError as e:
        print("Error sending message.", e)



def send_image(file, title):

    try:
        response = client.files_upload(
            channels="D01S9QSPVJN",
            file=file,
            title=title
        )

    except SlackApiError as e:
        print("Error sending image.", e)

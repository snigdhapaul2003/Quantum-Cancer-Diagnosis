import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
import mimetypes
import os
from email.message import EmailMessage
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def gmail_send_message(to,frm,subject,msg,creds,doc):
  """Create and send an email message
  Print the returned  message id
  Returns: Message object, including message id

  Load pre-authorized user credentials from the environment.
  for guides on implementing OAuth2 for the application.
  """

  try:
    service = build("gmail", "v1", credentials=creds)
    message = EmailMessage()

    message.set_content(msg)

    message["To"] = to
    message["From"] = frm
    message["Subject"] = subject
    attachment_filename = doc
    type_subtype, _ = mimetypes.guess_type(attachment_filename)
    maintype, subtype = type_subtype.split("/")
    with open(attachment_filename, "rb") as fp:
      attachment_data = fp.read()
    message.add_attachment(attachment_data, maintype, subtype, filename = attachment_filename)
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    create_message = {"raw": encoded_message}
    send_message = (
        service.users()
        .messages()
        .send(userId="me", body=create_message)
        .execute()
    )
    print(f'Message Id: {send_message["id"]}')
  except HttpError as error:
    print(f"An error occurred: {error}")
    send_message = None
  return send_message

def gmail_send(to, frm, subject, msg):
  """Shows basic usage of the Gmail API.
  Lists the user's Gmail labels.
  """
  creds = None
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    doc = "generated_report.pdf"
    gmail_send_message(to, frm, subject, msg, creds, doc)

  except HttpError as error:
    print(f"An error occurred: {error}")


if __name__ == "__main__":
  gmail_send()
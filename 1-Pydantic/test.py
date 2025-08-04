'''
Hackr.io Python Tutorial: Log Notification System
'''
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = "recipient_email@gmail.com"

# Function to send email


def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Define Event Handler


class LogFileHandler(FileSystemEventHandler):
    def __init__(self, keywords, file_path):
        self.keywords = keywords
        self.file_path = file_path

    def on_modified(self, event):
        if event.src_path == self.file_path:
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
                for line in lines[-10:]:  # Check only the last 10 lines for efficiency
                    for keyword in self.keywords:
                        if keyword in line:
                            subject = f"Alert: {keyword} detected"
                            body = f"Keyword '{keyword}' detected in log:\n\n{line}"
                            send_email(subject, body)


# Main Function
if __name__ == "__main__":
    path_to_watch = "logs/mylog.log"  # Replace with your log file
    keywords_to_watch = ["ERROR", "CRITICAL", "500 Internal Server Error"]

    event_handler = LogFileHandler(keywords_to_watch, path_to_watch)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)

    print("Monitoring started...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
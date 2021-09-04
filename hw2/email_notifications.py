import smtplib 
from email.message import EmailMessage
import imghdr

# WOW BOT - throwaway gmail notification account
WOW_BOT_EMAIL = "woolworths.bot001@gmail.com"
WOW_BOT_PASS = "%JKg5rtwoPZ*v&fUxe8U"

def send_email_notification(email_recipient, subject="", message_body="", fpath_image=None):
    """
    Overview
        Send email notification using throwaway gmail account
    Arguments    
        email_recipient - tuple of strings with email recipients ('alau3@woolworths.com.au', 'wxu1@woolworths.com.au')
        subject - email subject, string
        message_body - email message body, string
        fpath_image - filepath to an image file (if you want an image attaqched), string, optional
    """
    # create message
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = WOW_BOT_EMAIL
    msg['To'] = email_recipient
    msg.set_content(message_body)
    
    # attach image
    if fpath_image:
        with open(fpath_image, 'rb') as fp:
            img_data = fp.read()
        msg.add_attachment(img_data, maintype='image', subtype=imghdr.what(None, img_data))
    
    # send message
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp: #Added Gmails SMTP Server
        smtp.login(WOW_BOT_EMAIL, WOW_BOT_PASS) #This command Login SMTP Library using your GMAIL
        smtp.send_message(msg) #This Sends the message

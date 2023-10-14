import mailtrap as mt
import base64
from pathlib import Path
import os
import yaml

def read_token_yaml(path_yaml='../../mailtrap/mailtrap.yaml'):
    """Read token from yaml file
        username = mt_config['Username']
        token = mt_config['Password']
        port = mt_config['Port'][0]  
        sender = mt_config['Sender']   
    """
    with open(path_yaml, 'r') as stream:
        try:
            mt_config = yaml.safe_load(stream)                   
        except yaml.YAMLError as exc:
            print(exc)
    return mt_config

def send_mail_with_image(mt_config, 
                         receiver, 
                         subject="ai-surv detection", 
                         name="ai-urv",
                         text=None,
                         path_image=None):
    """Send email with image attached using mailtrap server"""
    
    # create mail object
    #
    # Image attachment
    if path_image is not None:
        name_image = os.path.basename(path_image)
        image = Path(path_image).read_bytes()
        attachments=[
            mt.Attachment(
                content=base64.b64encode(image),
                filename=name_image,
                disposition=mt.Disposition.INLINE,
                mimetype="image/png",
                content_id=name_image,
            )
        ]
    else:
        attachments=[None]

    mail = mt.Mail(
        sender=mt.Address(email=mt_config['sender'],
                          name=name), 
        to=[mt.Address(email=receiver)],
        # cc=[mt.Address(email="cc@examplecom")],
        # bcc=[mt.Address(email="bcc@examplecom")],
        subject=subject,
        text=text,
        # html="<html><body><h1>Mailtrap HTML</h1></body></html>",
        attachments=attachments,
    )

    # create client and send
    client = mt.MailtrapClient(token=mt_config['token'])
    client.send(mail)

def print_classes_found(detection_results):
    """Print classes and probabilities found in detection"""
    if len(detection_results) > 0:
        str_out = "Classes found: \n"
        for class_found, prob, _ in detection_results:
            str_out += f"\t-{class_found}, p={prob:.2f}. \n"
        print(str_out)
    else:
        print("No classes found")
    return str_out
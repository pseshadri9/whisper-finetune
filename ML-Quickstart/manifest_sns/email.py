import functools
import smtplib
import ssl
import logging
from email.message import EmailMessage
from .email_args import SOURCE_EMAIL, PASSWORD, DESTINATION_EMAIL, HEADER, SUBJECT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def SNS(subject=SUBJECT, header=HEADER):
    """
    A decorator that sends an email with metrics after the decorated function is called.
    
    Args:
        subject (str, optional): Custom email subject. Defaults to SUBJECT from email_args.
        header (str, optional): Custom email header text. Defaults to HEADER from email_args.
    
    Returns:
        callable: A decorator function that wraps the original function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the original function
            result = func(*args, **kwargs)
            
            # If the result is a dictionary, send an email with the metrics
            if isinstance(result, dict):
                if any([x == '' for x in (SOURCE_EMAIL, PASSWORD, DESTINATION_EMAIL)]):
                    log.info('ONE OR MORE REQUIRED ARGUMENTS MISSING FOR SNS, SKIPPING')
                else:
                    log.info('SENDING RESULTS TO SNS')
                    try:
                        send_email(args=result, subject=subject, header=header)
                    except Exception as e:
                        log.error(f'SKIPPING SNS, Exception occurred: {e}')
            
            return result
        
        return wrapper
    
    return decorator

def send_email(args: dict = None, subject: str = SUBJECT, header: str = HEADER):
    """
    Function to send notification email from SOURCE_EMAIL to DESTINATION_EMAIL
    
    Args:
        args (dict, optional): Dictionary of metrics. Defaults to None.
        subject (str, optional): Email subject. Defaults to SUBJECT from email_args.
        header (str, optional): Email header text. Defaults to HEADER from email_args.
    """
    context = ssl.create_default_context()
    msg = EmailMessage()
    msg.set_content(f'{header}\n{dict_to_str(args)}')

    msg['Subject'] = subject
    msg['From'] = SOURCE_EMAIL
    msg['To'] = DESTINATION_EMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SOURCE_EMAIL, PASSWORD)
        server.send_message(msg)

def dict_to_str(d):
    """
    Convert a dictionary to a formatted string.
    
    Args:
        d (dict): Input dictionary
    
    Returns:
        str: Formatted string representation of the dictionary
    """
    return '' if not d else '\n'.join([f'{k}: {v}' for k,v in d.items()])
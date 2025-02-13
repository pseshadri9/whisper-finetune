'''
SERIOUS: MAKE SURE THIS FILE IS IN GIT IGNORE!!!!!
DO NOT COMMIT SENSITIVE INFORMATION
'''

#Check that you read the above warning
import subprocess
list_of_ignored_files = str(subprocess.check_output("git ls-files --other", shell=True))

msg = f"""Failed Test!!!! Add \"logging/email_args.py\" to .gitignore to avoid committing sensitive information!!"""

assert ('email_args.py' in list_of_ignored_files), msg

SOURCE_EMAIL = ''
PASSWORD = ''
DESTINATION_EMAIL = ''
HEADER = ''
SUBJECT = ''
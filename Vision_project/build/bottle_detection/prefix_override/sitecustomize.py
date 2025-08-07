import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rupendra/Desktop/Tasks/Vision_project/install/bottle_detection'

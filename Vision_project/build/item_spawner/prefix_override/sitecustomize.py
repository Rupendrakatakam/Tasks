import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/humble/Desktop/Tasks/Vision_project/install/item_spawner'

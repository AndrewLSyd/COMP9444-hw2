import os
import shutil
import datetime
import time
now = datetime.datetime.now() + datetime.timedelta(hours=10)  # convert to Sydney time
print("Start training at", now.strftime("%Y-%m-%d %H:%M:%S"))
start_time = time.time()

source = 'student.py'
target = 'models/student_' + now.strftime("%Y-%m-%d_%H%M") + '.py'

source = 'test.py'
target = 'submissions'

target = os.path.join(target, os.path.dirname(source))

shutil.copy(source, target)

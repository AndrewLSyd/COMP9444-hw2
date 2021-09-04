#!bin/bash
# submit model
# usage

# 96.57%, 96.43%, 96.29%
# bash submit.sh checkModel_2021-08-04_1128_epoch_2580.pth student_2021-08-04_1128.py


# 96.71%...96.71%, 96.86%
# bash submit.sh checkModel_2021-08-03_1502_epoch_2260.pth student_2021-08-03_1502.py

# 95.86%
# bash submit.sh checkModel_2021-08-04_0011_epoch_2300.pth student_2021-08-04_0011.py


# 93.57%
# bash submit.sh checkModel_2021-08-02_1728_epoch_2260.pth student_2021-08-02_1728.py

# bash submit.sh checkModel_2021-08-01_1353_epoch_2200.pth student_2021-08-01_1353.py
# bash submit.sh checkModel_2021-08-01_1353_epoch_3000.pth student_2021-08-01_1353.py
# bash submit.sh checkModel_2021-08-02_1033_epoch_1260.pth student_2021-08-02_1033.py

sudo apt-get install sshpass

echo ">>> making copy in submissions folder"
cp models/$1 submissions/
cp models/$2 submissions/

echo ">>> making copy in submissions folder (with generic names)"
cp models/$1 submissions/savedModel.pth
cp models/$2 submissions/student.py

echo ">>> copying file over"
cd submissions
sshpass -p "baL#@6yVh#dx5na" scp student.py savedModel.pth z3330164@login.cse.unsw.edu.au:/import/ravel/2/z3330164/COMP9444/hw2

echo ">>> SSH in"
sshpass -f <(printf '%s\n' baL#@6yVh#dx5na) ssh z3330164@login.cse.unsw.edu.au 'cd COMP9444/hw2; give cs9444 hw2 student.py savedModel.pth'

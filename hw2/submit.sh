#!bin/bash
# submit model

sudo apt-get install sshpass

echo ">>> making copy in submissions folder"
cp models/$1.pth submissions/
cp student.py submissions/$1.py

cp models/$1.pth savedModel.pth

echo ">>> copying file over"
sshpass -p "baL#@6yVh#dx5na" scp student.py savedModel.pth z3330164@login.cse.unsw.edu.au:/import/ravel/2/z3330164/COMP9444/hw2

echo ">>> SSH in"
sshpass -f <(printf '%s\n' baL#@6yVh#dx5na) ssh z3330164@login.cse.unsw.edu.au 'cd COMP9444/hw2; give cs9444 hw2 student.py savedModel.pth'

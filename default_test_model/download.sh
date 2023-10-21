#!/bin/bash
fileid="1-D1VtBu7ik-jOm__FB3PdArFaTdVjt_Z"
filename="checkpoint.pth"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
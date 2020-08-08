#!/bin/bash

dataset=$1

function gdrive-get() {
    fileid=$1
    filename=$2
    if [[ "${fileid}" == "" || "${filename}" == "" ]]; then
        echo "gdrive-curl gdrive-url|gdrive-fileid filename"
        return 1
    else
        if [[ ${fileid} = http* ]]; then
            fileid=$(echo ${fileid} | sed "s/http.*drive.google.com.*id=\([^&]*\).*/\1/")
        fi
        echo "Download ${filename} from google drive with id ${fileid}..."
        cookie="/tmp/cookies.txt"
        curl -c ${cookie} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        confirmid=$(awk '/download/ {print $NF}' ${cookie})
        curl -Lb ${cookie} "https://drive.google.com/uc?export=download&confirm=${confirmid}&id=${fileid}" -o ${filename}
        rm -rf ${cookie}
        return 0
    fi
}

if [ ${dataset} == 'Eurlex-4K' ]; then
	gdrive-get 1pUBkYEvX_WBPApNQvFyqR3Px96TTwX2Y ${dataset}.tar.bz2
elif [ ${dataset} == 'Wiki10-31K' ]; then
	gdrive-get 1u8mOmPODKJVBYCxOSA6Gd9LPHIW61UAg ${dataset}.tar.bz2
elif [ ${dataset} == 'AmazonCat-13K' ]; then
	gdrive-get 1laaPUDOPyP59HnRS3oMQul5gZ5pVEpIs ${dataset}.tar.bz2
elif [ ${dataset} == 'Wiki-500K' ]; then
	gdrive-get 1ZgVYpfZ_P_sl89o9edbVLZwGS4ieBZ0Q ${dataset}.tar.bz2
else
	echo "unknown dataset [ Eurlex-4K | Wiki10-31K | AmazonCat-13K | Wiki-500K ]"
	exit
fi

tar -xjvf ${dataset}.tar.bz2


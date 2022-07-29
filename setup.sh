#!/bin/bash
ZIP_FILE_WITHOUT_EXT="cil-road-segmentation-2022"
mkdir -p $ZIP_FILE_WITHOUT_EXT
wget "https://polybox.ethz.ch/index.php/s/yqbZH3QszgkGrDh/download" && mv download $ZIP_FILE_WITHOUT_EXT.zip
if unzip $ZIP_FILE_WITHOUT_EXT.zip -d $ZIP_FILE_WITHOUT_EXT
then
    mkdir -p data/split/test/images data/split/test/groundtruth
    mv $ZIP_FILE_WITHOUT_EXT/test data/
    mv $ZIP_FILE_WITHOUT_EXT/training/ data/split/train/
    val=$(cat validation-files-list.dat | sed -n 's/- satimage/satimage/p')
    for file in $val
    do
        mv data/split/train/images/$file data/split/test/images/
        mv data/split/train/groundtruth/$file data/split/test/groundtruth/
    done
    
    echo -e "Will download the checkpoint.\nPress enter to continue"
    read
    wget "https://polybox.ethz.ch/index.php/s/1fAbWrYUuf3oLWP/download" && mv download globe.ckpt
    
    echo -e 'Will upgrade pip and install the requirements. If you want them installed in e.g. a venv, make sure to activate the venv before running the script AND run the script like `. ./setup.sh` (not just `./setup.sh`) to have the venv active'
    echo "Press enter to continue."
    read
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
else
    echo "Please download the dataset zip and store it in this directory under the name cil-road-segmentation-2022.zip"
fi

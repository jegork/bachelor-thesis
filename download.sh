apt-get update
apt-get install unzip unrar
wget --continue --progress=dot:mega --tries=0 https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
unzip UCF*.zip
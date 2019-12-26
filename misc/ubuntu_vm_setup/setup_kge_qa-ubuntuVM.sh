# This guide describes how to setup the KGE QA system on Ubuntu VM, 
# using a (fresh) new `ubuntu-basic.ova` Virtual Machine Environment (Dr. Tao's version)
# The VM image downloadable here: https://www.dropbox.com/sh/714sglbrw1jc4la/AACsBIrXfxASZ_EZbqXHz5Nva?dl=0
# 
# 
# KGE QA: "Factoid-Questions Answering System based on Knowledge Graph"
# By: Aziz Altowayan
# 

echo When prompted, ENTER Ubuntu PASSWORD
echo e.g. [sudo] password for pace: 123456


### Install helper software for setting up Ubuntu
echo -------------------------------------------------------
echo -------------------------------------------------------
echo Setting up the Ubuntu environment ...
echo -------------------------------------------------------
echo -------------------------------------------------------
sudo apt-get update -y
sudo apt -y install curl
sudo apt-get -y install python3
sudo apt-get -y install python3-pip

### Download KGE QA source code
echo -------------------------------------------------------
echo -------------------------------------------------------
echo Downloading source code files ...
echo -------------------------------------------------------
echo -------------------------------------------------------
cd ~/Desktop/
wget -L https://www.dropbox.com/s/oi9gylxyw0jkwef/kge_qa.tar.gz
echo extracting the source files ...
tar xzf kge_qa.tar.gz
cd kge_qa
echo -------------------------------------------------------
echo -------------------------------------------------------
echo Downloading the pre-trained embedding models ...
echo -------------------------------------------------------
echo -------------------------------------------------------
curl http://magnitude.plasticity.ai/fasttext/medium/wiki-news-300d-1M-subword.magnitude -o lab/datasets/ft-wiki-news-300d-1M-subword.magnitude

### Install KGE_QA dependencies
echo -------------------------------------------------------
echo -------------------------------------------------------
echo Installing the QA system dependencies ...
echo -------------------------------------------------------
echo -------------------------------------------------------
pip3 uninstall -y numpy # run repeatedly to uninstall all the older numpy versions (due to version-conflict with streamlit)
pip3 uninstall -y numpy
pip3 uninstall -y numpy
pip3 install -r requirements.txt


echo -------------------------------------------------------
echo -------------------------------------------------------
echo SETUP IS COMPLETED
echo -------------------------------------------------------
echo -------------------------------------------------------
### Launch the KGE_QA system
echo To start KGE_QA UI app, run the file: "run_kge_qa-app.sh"
# PATH="$HOME/.local/bin/:$PATH"
# streamlit run app.py
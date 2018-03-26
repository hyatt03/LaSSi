#! /bin/bash

# Install basic mccode dependencies

sudo yum update
sudo yum -y upgrade
sudo yum install -y yum-utils git
sudo yum -y groupinstall development
sudo yum -y install https://centos7.iuscommunity.org/ius-release.rpm
sudo yum -y install python36u python36u-devel python36u-pip
sudo pip3.6 install numpy scipy pandas ase matplotlib tables

cd /home/vagrant

sudo -u vagrant git clone https://github.com/hyatt03/b1p.git

cd b1p

FROM python:3.6-slim

# COPY . /app

RUN apt-get -y update && apt-get -y install git && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/man/?? /usr/share/man/??_*
RUN git clone https://github.com/nikitakrutoy/nngrid /setup/nngrid && python /setup/nngrid/setup.py install

RUN nngrid clone https://github.com/nikitakrutoy/nngird-sample-project.git 
RUN nngrid worker nikitakrutoy-nngird-sample-project
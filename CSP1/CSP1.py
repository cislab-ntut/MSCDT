import os
from entity2 import CloudServiceProvider1

import configparser

config = configparser.ConfigParser()
config.read('MSCDT.cfg')

r_config = configparser.ConfigParser()
r_config.set('DEFAULT', 'CSP0_SEVER_IP', os.environ.get('CSP0_SEVER_IP', config['DEFAULT']['CSP0_SEVER_IP']))
r_config.set('DEFAULT', 'CSP1_SEVER_IP', os.environ.get('CSP1_SEVER_IP', config['DEFAULT']['CSP1_SEVER_IP']))
r_config.set('DEFAULT', 'CSP0_SEVER_PORT', os.environ.get('CSP0_SEVER_PORT', config['DEFAULT']['CSP0_SEVER_PORT']))
r_config.set('DEFAULT', 'CSP1_SEVER_PORT', os.environ.get('CSP1_SEVER_PORT', config['DEFAULT']['CSP1_SEVER_PORT']))
r_config.set('DEFAULT', 'RETRIES', os.environ.get('RETRIES', config['DEFAULT']['RETRIES']))

if __name__ == "__main__":
    csp1 = CloudServiceProvider1(r_config, version=3)
    csp1.start_connection()
    csp1.cspthread.join()
    # while True:
    #     input()
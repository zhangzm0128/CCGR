import os
import json
import argparse
import time
from network.network import *
from utils.data_loader import DataLoader
from utils.logger import LoggerWriter

from train import Train
from predict import Predict
from test import Test

# Load external parameters
parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='config.json',
                    help='the path of global config file.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='the path of checkpoint and program will run checkpoint data.')
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--mode', type=str, default='train',
                    help='the mode of app will run, plz choose among ["train", "test", "predict"]')
parser.add_argument('--device', type=str, default='cuda',
                    help='the device of app will run, plz choose among ["cuda", "cpu"]')

args = parser.parse_args()

model_name = args.model_name
checkpoint = args.checkpoint
mode = args.mode
device = args.device

if mode == 'train':
    config_file = open(args.config, 'r').read()
    config = json.loads(config_file)
else:
    config_in_checkpoint = os.path.join(checkpoint, 'config', 'config.json')
    config_file = open(config_in_checkpoint, 'r').read()
    config = json.loads(config_file)



# Select mode to run
if mode == 'train':
    net_name = config['network_params']['name']
    net = eval(net_name)(config['network_params'], device)
    print(net.parameters())
    
    train_data_loader = DataLoader(config['data_params'], 'train', 'memory')
    val_data_loader = DataLoader(config['data_params'], 'valid', 'memory')

    logger = LoggerWriter(config, checkpoint)
    logger.set_log_format('Epoch,Iter,Loss-MSE,Time\n')
    logger.init_logs()
    trainer = Train(config, logger, net, train_data_loader, val_data_loader, device)
    trainer.train()
elif mode == 'test':
    logger = LoggerWriter(config, checkpoint)
    net = logger.load_model(device=device, model_name=model_name)
    val_data_loader = DataLoader(config['data_params'], 'valid', 'memory')
    tester = Test(config, logger, net, val_data_loader, device)
    tester.test()
elif mode == 'predict':   
    logger = LoggerWriter(config, checkpoint)
    net = logger.load_model(device=device, model_name=model_name)
    pred_data_loader = DataLoader(config['data_params'], 'predict', 'memory')
    predictor = Predict(config, logger, net, pred_data_loader, device)
    predictor.predict()
else:
    print('Plz choose correct mode among ["train", "test", "predict"]')

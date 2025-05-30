import torch
import numpy as np
import pandas as pd
import argparse
import time
import os
import util
from util import *
import random
from CrossST_pre import CrossST_pre
import torch.optim as optim
from ranger21 import Ranger

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1", help="")
parser.add_argument("--data", type=str, default="pre_train", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--d_model", type=int, default=256, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=48, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=100, help="")
parser.add_argument("--print_every", type=int, default=100, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
parser.add_argument(
    "--es_patience",
    type=int,
    default=5,
    help="quit if no improvement after this many iterations",
)
args = parser.parse_args()

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(current_time)

def get_scaler(scaler_dict, x):
    num_nodes = x.size(2)

    num_nodes_to_dataset = {
        480: 'CAD3',
        2352: 'CAD4',
        211: 'CAD5',
        484: 'CAD6',
        1859: 'CAD7',
        1022: 'CAD8',
        523: 'CAD10',
        716: 'CAD11',
        953: 'CAD12',
    }
    dataset_name = num_nodes_to_dataset.get(num_nodes)
    
    if dataset_name is not None:
        return scaler_dict[dataset_name]
    else:
        raise ValueError(f"无法找到对应的数据集scaler，num_nodes: {num_nodes}")

class trainer:
    def __init__(
        self,
        scaler_dict,
        input_dim,
        d_model,
        num_nodes,
        input_len,
        output_len,
        dropout,
        lrate,
        wdecay,
        device,
    ):
        self.model = CrossST_pre(
            device, input_dim, d_model, input_len, output_len, dropout
        )
        self.scaler_dict = scaler_dict
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.clip = 5
        print("The number of parameters: {}".format(self.model.param_num()))
        print(self.model)
        
    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)  
        real = torch.unsqueeze(real_val, dim=-1)  
        scaler = get_scaler(self.scaler_dict, input)
        predict = scaler.inverse_transform(output)  
        real = scaler.inverse_transform(real)  
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        real = torch.unsqueeze(real_val, dim=-1)
        scaler = get_scaler(self.scaler_dict, input) 
        predict = scaler.inverse_transform(output)
        real = scaler.inverse_transform(real)  
        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape


def seed_it(seed):
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    seed_it(6666)

    data = args.data
    args.input_len = 96
    args.output_len = 96
    args.batch_size = 40
    
    device = torch.device(args.device)

    dataset_list = [
        'CAD4',
        'CAD7',
        'CAD8',
        'CAD11',
        'CAD12',
    ]

    # dataset_list = [
    #     'CAD11',
    # ]

    train_loader, valid_loader, test_loader, scaler_dict = util.load_pre_data_with_dataloader(
        dataset_list, args.batch_size, args.input_len, args.output_len, args.batch_size, args.batch_size
    )
    
    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + data + "/"

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(
        scaler_dict,
        args.input_dim,
        args.d_model,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
    )

    print("start training...", flush=True)

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        for iter, (x, y) in enumerate(train_loader):
            x = x.squeeze(0)
            y = y.squeeze(0)
            trainx = x.to(device)  
            trainy = y.to(device)
            metrics = engine.train(trainx, trainy[:, :, :, 0])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            if iter % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}"
                print(
                    log.format(
                        iter,
                        np.mean(train_loss),
                        np.mean(train_rmse),
                        np.mean(train_mape),
                        np.mean(train_wmape),
                    ),
                    flush=True,
                )
        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(valid_loader):
            x = x.squeeze(0)
            y = y.squeeze(0)
            testx = x.to(device)
            testy = y.to(device)
            metrics = engine.eval(testx, testy[:, :, :, 0])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(
            log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            bestid = i
            epochs_since_best_mae = 0
            print("Updating! Valid Loss:", mvalid_loss, end=", ")
            print("epoch: ", i)
        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        if epochs_since_best_mae >= args.es_patience and i >= 200:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

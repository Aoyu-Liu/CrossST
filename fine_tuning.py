import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from CrossST_pre import CrossST_pre as model_p
from CrossST_fine import CrossST_fine as model_f
import torch.optim as optim
from ranger21 import Ranger

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1", help="")
parser.add_argument("--data", type=str, default="CAD5_10", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--d_model", type=int, default=256, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=500, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
parser.add_argument(
    "--es_patience",
    type=int,
    default=50,
    help="quit if no improvement after this many iterations",
)
args = parser.parse_args()


class DynamicWeighting:
    def __init__(self):
        self.weights = [1.0, 1.0, 1.0]
    
    def update_weights(self, loss1, loss2, loss3):
        total_loss = (loss1 + loss2 + loss3)
        self.weights[0] = total_loss / loss1 if loss1 != 0 else 1.0
        self.weights[1] = total_loss / loss2 if loss2 != 0 else 1.0
        self.weights[2] = total_loss / loss3 if loss3 != 0 else 1.0


class trainer:
    def __init__(
        self,
        scaler,
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
        pretrained_weights_path = "./logs/pre_train_best_model.pth"

        print(pretrained_weights_path)

        self.model_p = model_p(
            device, input_dim, d_model, input_len, output_len, dropout, "fine_tuning"
        )
        self.model_p.to(device)
        
        pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
        self.model_p.load_state_dict(pretrained_dict, strict=False)
        
        for param in self.model_p.parameters():
            param.requires_grad = False
        print("权重导入成功！")

        self.model = model_f(
            device, input_dim, d_model, num_nodes, input_len, output_len, dropout, self.model_p
        )
        
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        kl = 0.3
        cl = 0.3
        self.kd = util.DistillKL(kl)
        self.cl =  util.InfoNCELoss(cl)
        print("k=",kl)
        print("c=",cl)
        self.dynamic_weighting = DynamicWeighting()
        self.scaler = scaler
        self.clip = 5
        print("The number of parameters: {}".format(self.model.param_num()))
        print(self.model)


    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output, t_h_t, t_h_s , s_h_t, s_h_s = self.model(input)  
        real = torch.unsqueeze(real_val, dim=-1)  
        predict = self.scaler.inverse_transform(output)  
        real = self.scaler.inverse_transform(real)  

        loss1 =self.loss(predict, real, 0.0) 
        loss2 =self.kd(s_h_t,t_h_t)
        loss3 =self.cl(t_h_s,s_h_s)
        
        self.dynamic_weighting.update_weights(loss1, loss2, loss3)
        loss = self.dynamic_weighting.weights[0]*loss1*0.8 + 0.2*(self.dynamic_weighting.weights[1]*loss2+ self.dynamic_weighting.weights[2]*loss3)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.loss(predict, real, 0.0).item()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return mae, mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output, t_h_t, t_h_s , s_h_t, s_h_s = self.model(input)
        real = torch.unsqueeze(real_val, dim=-1)
        predict = self.scaler.inverse_transform(output)
        real = self.scaler.inverse_transform(real)  
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

    if args.data == "CAD3":
        args.data = "./val_data/" + args.data
        args.num_nodes = 480
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD5":
        args.data = "./val_data/" + args.data
        args.num_nodes = 211
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD6":
        args.data = "./val_data/" + args.data
        args.num_nodes = 484
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD10":
        args.data = "./val_data/" + args.data
        args.num_nodes = 523
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD3_10":
        args.data = "./val_data/" + args.data
        args.num_nodes = 480
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD5_10":
        args.data = "./val_data/" + args.data
        args.num_nodes = 211
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD6_10":
        args.data = "./val_data/" + args.data
        args.num_nodes = 484
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD10_10":
        args.data = "./val_data/" + args.data
        args.num_nodes = 523
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD3_st_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 480
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD5_st_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 211
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD6_st_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 484
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD10_st_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 523
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD3_st_mask20":
        args.data = "./val_data/" + args.data
        args.num_nodes = 480
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD5_st_mask20":
        args.data = "./val_data/" + args.data
        args.num_nodes = 211
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD6_st_mask20":
        args.data = "./val_data/" + args.data
        args.num_nodes = 484
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD10_st_mask20":
        args.data = "./val_data/" + args.data
        args.num_nodes = 523
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD3_s_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 480
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD5_s_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 211
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD6_s_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 484
        args.input_len = 96
        args.output_len = 96

    elif args.data == "CAD10_s_mask30":
        args.data = "./val_data/" + args.data
        args.num_nodes = 523
        args.input_len = 96
        args.output_len = 96


    args.save = "./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-"
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(current_time)

 
    device = torch.device(args.device)

    train_loader, valid_loader, test_loader, scaler = util.load_fine_data_with_dataloader(
        args.data, args.batch_size, args.input_len, args.output_len, args.batch_size, args.batch_size,
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
        scaler,
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

    print("start training...")

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        for iter, (x, y) in enumerate(train_loader):
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
                        train_loss[-1],
                        train_rmse[-1],
                        train_mape[-1],
                        train_wmape[-1],
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
            if i < 50:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mvalid_loss, end=", ")
                print("epoch: ", i)

            elif i > 50:
                outputs = []
                test_labels = []

                for iter, (x, y) in enumerate(test_loader):
                    testx = x.to(device)
                    with torch.no_grad():
                        preds, t_h_t, t_h_s , s_h_t, s_h_s = engine.model(testx)
                    outputs.append(preds.squeeze())
                    test_labels.append(y.squeeze())

                realy = torch.cat(test_labels, dim=0).to(device)

                yhat = torch.cat(outputs, dim=0)
                yhat = yhat[: realy.size(0), ...]

                amae = []
                amape = []
                awmape = []
                armse = []
                test_m = []

                for j in range(args.output_len):
                    pred = scaler.inverse_transform(yhat[:, j, :])
                    real = scaler.inverse_transform(realy[:, j, :])
                    metrics = util.metric(pred, real)
                    log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                    print(
                        log.format(
                            j + 1, metrics[0], metrics[2], metrics[1], metrics[3]
                        )
                    )

                    test_m = dict(
                        test_loss=np.mean(metrics[0]),
                        test_rmse=np.mean(metrics[2]),
                        test_mape=np.mean(metrics[1]),
                        test_wmape=np.mean(metrics[3]),
                    )
                    test_m = pd.Series(test_m)

                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])
                    awmape.append(metrics[3])

                log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                print(
                    log.format(
                        np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)
                    )
                )

                if np.mean(amae) < test_log:
                    test_log = np.mean(amae)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss:", np.mean(amae), end=", ")
                    print("Test low! Updating! Valid Loss:", mvalid_loss, end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        if epochs_since_best_mae >= args.es_patience and i >= 50:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))

    outputs = []
    test_labels = []

    for iter, (x, y) in enumerate(test_loader):
        testx = x.to(device)
        with torch.no_grad():
           preds, t_h_t, t_h_s, h_t, h_s = engine.model(testx)
        outputs.append(preds.squeeze())
        test_labels.append(y.squeeze())

    realy = torch.cat(test_labels, dim=0).to(device)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]


    amae = []
    amape = []
    armse = []
    awmape = []
    test_m = []
    
    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, i, :])
        real = scaler.inverse_transform(realy[:, i, :])
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

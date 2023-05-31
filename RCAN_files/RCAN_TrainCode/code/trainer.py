import os
import math
from decimal import Decimal

import utility
import symbacdata

import torch
from torch.autograd import Variable
from tqdm import tqdm

from piqa import SSIM, PSNR
import tifffile
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from imageio import imsave

def open_image(filename):
    if "tif" in filename:
        img = np.array(tifffile.imread(filename),dtype=np.float32)
    elif "png" in filename:
        img = np.array(Image.open(filename),dtype=np.float32)
    else:
        raise TypeError(f"Images must be tif files or png files, not {filename[-3:]}")
    return img

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.ssim = SSIM(n_channels=1, value_range=255.0)
        self.psnr = PSNR(value_range=255.0)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): 
                self.optimizer.step()
                self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr() #get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, lr[0])#Decimal(lr))
        )
        self.ckp.write_ssim(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, lr[0])#Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
 
        timer_data, timer_model = utility.timer(), utility.timer()
        # print(f"length of loader train is: {len(self.loader_train)}")
        # print(f"the batch size is {self.args.batch_size}, in my opinion")
        inds = np.sort(np.random.choice(len(self.loader_train),size=50,replace=False))
        # print(f"inds is {inds}")
        batch = 0
        for idx, (lr, hr) in enumerate(self.loader_train):
            if idx not in inds:
                continue
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                print(f"the current batch is {batch}")
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            batch += 1

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        # print(f"THE FAST ONE IS: {epoch}")
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.ckp.add_ssim(torch.zeros(1, len(self.scale)))
        self.model.eval()

        if self.args.sr_dir_src != ".":
            src_dir = self.args.sr_dir_src
            if not os.path.exists(src_dir): raise FileNotFoundError(f"{src_dir} not found")
            hr_dir = self.args.sr_dir_hr
            if not os.path.exists(hr_dir): raise FileNotFoundError(f"{hr_dir} not found")
            dst_dir = self.args.sr_dir_dst
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir) 
                print(f"made directory: {dst_dir}")
            else:
                print(f"directory already exists: {dst_dir}")
            eval_dataset, eval_dataset1 = symbacdata.get_datasets(src_dir, hr_dir, src_dir, hr_dir)
            loader = symbacdata.get_dataloaders(eval_dataset, eval_dataset)[1]
            filenames = os.listdir(src_dir)
            inf = True # using the model to do inference
        else:
            loader = self.loader_test
            inf = False # continuing training on a model

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                ssim_acc = 0
                # self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(loader, ncols=80)
                for idx_img, (lr, hr) in enumerate(tqdm_test):
                    if not inf:
                        filename = f"epoch_{epoch}_img_{idx_img}.png"
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    
                    #doing inference
                    if inf:
                        imsave(f"{dst_dir}/{filenames[idx_img]}",np.array(sr.cpu(),dtype=np.uint8).reshape(207,54))
                        continue
                    # print("saving sr image")

                    # imsave("sr.png",np.array(sr.cpu()).reshape(207,54))
                    # print("saving lr image")
                    # imsave("lr.png",np.array(lr.cpu()).reshape(207,54))
                    # print("saving hr image")
                    # imsave("hr.png",np.array(hr.cpu()).reshape(207,54))
                    
                    save_list = [sr]
                    if not no_eval:
                        # print(f"device sr: {sr.device}, device hr: {hr.device}")
                        eval_acc += self.psnr(sr, hr)
                        ssim_acc += self.ssim(sr.cuda(), hr.cuda())
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                self.ckp.ssim[-1] = ssim_acc / len(self.loader_test)
                best_psnr = self.ckp.log.max(0)
                best_ssim = self.ckp.ssim.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best_psnr[0][idx_scale],
                        best_psnr[1][idx_scale] + 1
                    )
                )
                # print("--------------------")
                # print(f"{self.args.data_test},{type(self.args.data_test)}")
                # print(f"{scale},{type(scale)}")
                # print(f"{self.ckp.ssim[-1, idx_scale]},{type(self.ckp.ssim[-1, idx_scale])}")
                # print(f"{best_ssim[0][idx_scale]},{type(best_ssim[0][idx_scale])}")
                # print(f"{best_ssim[1][idx_scale] + 1},{type(best_ssim[1][idx_scale] + 1)}")
                # print("--------------------")
                self.ckp.write_ssim(
                    '[{} x{}]\tSSIM: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.ssim[-1, idx_scale],
                        best_ssim[0][idx_scale],
                        best_ssim[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        self.ckp.write_ssim(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            print(f"THE EPOCH IS: {epoch}")
            self.ckp.save(self, epoch, is_best=(best_psnr[1][0] + 1 == epoch))
            self.ckp.save(self, epoch, is_best=(best_ssim[1][0] + 1 == epoch))

        # else:
        #     src_dir = self.args.sr_dir_src
        #     if not os.path.exists(src_dir): raise FileNotFoundError(f"{src_dir} not found")
        #     dst_dir = self.args.sr_dir_dst
        #     if not os.path.exists(dst_dir): 
        #         os.mkdir(dst_dir) 
        #         print(f"made directory: {dst_dir}")
        #     else:
        #         print(f"directory already exists: {dst_dir}")
        #     for file in tqdm(os.listdir(src_dir)):
        #         lr = open_image(src_dir+file)
        #         lr = lr.reshape(1,1,207,54)
        #         lr = (lr//256) 
        #         lr = torch.tensor(lr, dtype=torch.float)
        #         lr = lr.to(torch.device("cuda"))
        #         # print(f"the device here is {upsampled.device}")
        #         with torch.no_grad():
        #             output = self.model(lr,1).cpu()
        #             sr_image = (np.array(output).reshape(207,54)).astype(int)
        #         sr_image = Image.fromarray(sr_image).convert("L")
        #         sr_image.save(dst_dir+file[:-3]+"png")


    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs + 1


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from utils.common_utils import *
import argparse
from datasets.NaturalColor import NaturalColor
from datetime import datetime
from utils.dip_utils import loss_fn, set_seed
from models import *
from models.downsampler import Downsampler
from utils.os_utils_dip import logs_folder_structure, save_outputs
from utils.quality_measures import compute_qlt_measures
from utils.results_saver import results_saver
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.torch_utils import get_transforms, get_opt, PSNR
from utils.os_utils import get_identifier, remove_files_in_directory

print(os.getenv('DISPLAY'))
dtype = torch.cuda.FloatTensor


# Set same seed for everything
set_seed()


def sr_dip(patch: dict,
           method: int,
           n_its: int,
           reg_noise_std: float,
           lr: float = 0.001,
           input_depth: int = 32,
           dataset: str = "",
           identifier: str = None,
           device: str = "cuda",
           tv_param: float = 0,
           nn_param: float = 0,
           ssahtv_param: float = 0,
           schiavi_param: float = 0,
           p_tv: float = 0,
           p_nn: float = 0,
           alpha: float = 0,
           beta: float = 0,
           factor: int = 4,
           # ema_param: float=0.99
           ):

    assert method in [0, 1, 2], "The method is not valid"

    log_dir = logs_folder_structure(main_folder="sr", dataset=dataset, identifier=identifier)

    writer: torch.utils.tensorboard.SummaryWriter = SummaryWriter(log_dir)

    if len(patch['y'].size()) != 4:
        gt = patch['y'][None, ...]
    else:
        gt = patch['y']

    gt = gt.to(device)

    b, c, h, w = gt.size()

    print("Original patch size: ", b, c, h, w)

    # Get the low resolution image

    low_res = F.interpolate(gt, scale_factor=1/factor, mode='bicubic', align_corners=False)

    INPUT = "noise"
    pad = "reflection"
    NET_TYPE = "skip"
    KERNEL_TYPE = "lanczos2"

    net_input = get_noise(input_depth=input_depth, method=INPUT, spatial_size=(h, w)).type(dtype).detach().to(device)
    net_input_saved = net_input.clone()

    net = get_net(input_depth, NET_TYPE, pad,
                  num_output_channels=c,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype).to(device)

    opt_theta = get_opt(model=net, opt="adam", lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_theta, T_max=(n_its - 1), eta_min=1e-2*lr)

    if method == 0:
        dw = Downsampler(n_planes=c, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)
        dw = [dw]
    elif method == 1 or method == 2:
        dw1 = Downsampler(n_planes=c, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)
        dw2 = Downsampler(n_planes=c, factor=factor, kernel_type="lanczos3", phase=0.5, preserve_size=True).type(dtype)
        dw = [dw1, dw2]

    best: list = [None, 0]
    psnr_fn = PSNR(max=1)

    hrEMA = None

    for it in tqdm(range(n_its)):
        net_input = net_input_saved + reg_noise_std * torch.randn_like(net_input_saved)
        # low_res_ = low_res + 0.01*torch.randn_like(low_res)  # TODO: ANALIZAR esto
        low_res_ = low_res

        opt_theta.zero_grad()
        hr = net(net_input)
        loss = loss_fn(hr=hr, low_res=low_res_, dw=dw, method=method, tv_param=tv_param, nn_param=nn_param,
                       ssahtv_param=ssahtv_param, schiavi_param=schiavi_param, p_tv=p_tv, p_nn=p_nn, alpha=alpha,
                       beta=beta)

        # if hrEMA is None:
        #     hrEMA = hr
        # else:
        #     hrEMA = hrEMA * ema_param + hr * (1 - ema_param)

        loss.backward()
        opt_theta.step()
        # scheduler.step()

        if it % 50 == 0:
            org_lr = torchvision.transforms.Resize(size=(h, w),
                                                   interpolation=torchvision.transforms.InterpolationMode.NEAREST)(low_res)

            gt_ = gt[0, :, ...]
            hr_ = hr[0, :, ...]
            # hr_ = hrEMA[0, :, ...]
            lr_ = org_lr[0, :, ...]
            dif = 10*torch.abs(gt_ - hr_)

            all_ = torch.concat([lr_, hr_, dif, gt_], dim=-1)

            writer.add_image('LR, HR, DIF, GT', all_, global_step=it)


        mae_hr = torch.abs(hr - gt).mean().item()
        mse_hr = torch.square(hr - gt).mean().item()
        psnr = psnr_fn(hr, gt)
        # psnr = psnr_fn(hrEMA, gt)

        writer.add_scalar("iteration", it, global_step=it)
        writer.add_scalar("psnr", psnr, global_step=it)
        writer.add_scalar("mae_hr", mae_hr, global_step=it)
        writer.add_scalar("mse_hr", mse_hr, global_step=it)
        writer.add_scalar("params_net_loss", loss.item(), global_step=it)

        checkpoint = {
                      "iteration": it,
                      "net_st_dict": net.state_dict(),
        }

        if best[0] is None:
            best[0] = psnr
            best[1] = it

        if best[0] is not None and psnr > best[0]:
            best[0] = psnr
            best[1] = it

        remove_files_in_directory(log_dir + "/checkpoints/")

        torch.save(checkpoint, log_dir + "/checkpoints/" + "iteration-" + str(it) + ".pth")

    print('MAE HR: {:.6f}, Loss: {:.6f}'.format(mae_hr, loss.item()))
    print("\n")
    print(best)

    return net(net_input), gt, mae_hr, loss.item(), best, writer


def main():

    print(f'Using GPU: {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Description of my script')
    parser.add_argument('--method', type=int, default=1, help='Method for downsampling')
    parser.add_argument('--n_its', type=int, default=5500, help='Number of iterations')
    parser.add_argument('--reg_noise_std', type=float, default=0.01, help='Standard deviation of noise')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')  # 0.01
    parser.add_argument('--dataset', type=str, default='KODAK', help='Dataset')
    parser.add_argument('--input_depth', type=int, default=32, help='Depth of input noise tensor')
    parser.add_argument('--tv_param', type=float, default=0, help='Weighing of tv regularization')
    parser.add_argument('--nn_param', type=float, default=0, help='Weighing of nuclear norm regularization')
    parser.add_argument('--ssahtv_param', type=float, default=0.8, help='Weighing of sssahtv regularization')
    parser.add_argument('--schiavi_param', type=float, default=0, help='Weighing of schiavi regularization')
    parser.add_argument('--p_tv', type=float, default=1, help='p-value for TV')
    parser.add_argument('--p_nn', type=float, default=1, help='p-value for NN')
    parser.add_argument('--alpha', type=float, default=2.8, help='alpha value')
    parser.add_argument('--beta', type=float, default=0.35, help='beta value')
    parser.add_argument('--factor', type=float, default=2, help='downsampling factor')
    # parser.add_argument('--ema_param', type=float, default=0, help='downsampling factor')


    args = parser.parse_args()
    print(args)

    args_dict: dict = vars(args)

    # Get identifier of the experiment
    identifier: str = get_identifier(args=args_dict)

    # Get transforms
    composed, _, _ = get_transforms(dataset=args.dataset, precision=32)

    # Get dataset and dataloader
    dataset = NaturalColor(root_dir="DATA/" + args.dataset + '/', transform=composed)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    for i, patch in enumerate(dataloader):
        print("Processing patch number: ", i)

        init_time: str = datetime.now().strftime('%Y/%m/%d_%H:%M:%S')

        full_sr, gt, mae_hr, loss, best, writer = sr_dip(patch=patch,
                                                         method=args.method,
                                                         n_its=args.n_its,
                                                         reg_noise_std=args.reg_noise_std,
                                                         lr=args.lr,
                                                         input_depth=args.input_depth,
                                                         identifier=identifier,
                                                         device="cuda",
                                                         dataset=args.dataset,
                                                         tv_param=args.tv_param,
                                                         nn_param=args.nn_param,
                                                         ssahtv_param=args.ssahtv_param,
                                                         schiavi_param=args.schiavi_param,
                                                         p_tv=args.p_tv,
                                                         p_nn=args.p_nn,
                                                         alpha=args.alpha,
                                                         beta=args.beta,
                                                         factor=args.factor,
                                                         # ema_param=args.ema_param
                                                         )

        final_time: str = datetime.now().strftime('%Y/%m/%d_%H:%M:%S')

        qm_mae, qm_mse, qm_ssim, qm_psnr, qm_sam = compute_qlt_measures(infered=full_sr, gt=gt)
        print(qm_mae, qm_mse, qm_ssim, qm_psnr)

        # save_outputs(folder=args.dataset, infered=full_sr, gt=gt, args_dict=args_dict,
        #              dataset=args.dataset, identifier=identifier)

        args_results: dict = {"mae_hr": mae_hr, "loss": loss, "init_time": init_time, "final_time": final_time,
                              "qm_mae": qm_mae, "qm_mse": qm_mse, "qm_ssim": qm_ssim, "qm_psnr": qm_psnr,
                              "qm_sam": qm_sam, "number_patch": i, "best_psnr": best[0], "best_it": best[1]}

        results_saver(folder="results_experiments/", name_csv=args.dataset + ".csv", args_dict=args_dict,
                      args_results=args_results)



if __name__ == "__main__":
    main()

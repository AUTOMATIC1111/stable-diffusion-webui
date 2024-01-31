import argparse
def setting_data(data_name, parser):

    # Notice:
    # Correct file structure should be
    # -- data_dir
    #    |-- tra_img_dir
    #    |   |-- xxx1.jpg
    #    |   |-- xxx2.jpg
    #    |   |-- ...
    #    |-- tra_label_dir
    #    |   |-- xxx1.png
    #    |   |-- xxx2.png
    #    |   |-- ...
    #    |-- tra_edge_dir
    #    |-- test_img_dir
    #    |-- test_label_dir
    #    |-- pred_results_dir


    availiable_setting = ['DH', 'UH','KUH','HRSOD','HRS10K']

    if data_name not in availiable_setting:
        raise NotImplementedError('Change setting in ./config.py')

    if data_name == 'HRSOD':
        # HRSOD
        parser.add_argument('--data_dir', type=str,default='./train_data/HRSOD/', help='Parent folder')
        parser.add_argument('--tra_img_dir',type=str,default='HRSOD_train/',help='Location of training image')
        parser.add_argument('--tra_label_dir',type=str,default='HRSOD_train_mask/',help='train label')
        parser.add_argument('--tra_edge_dir',type=str,default='HRSOD_train_edge/',help='train label')

    elif data_name == 'HRS10K':
        # HR10K
        parser.add_argument('--data_dir', type=str,default='./train_data/HR10K/train/', help='Parent folder')
        parser.add_argument('--tra_img_dir',type=str,default='img_train_2560max/',help='Location of training image')
        parser.add_argument('--tra_label_dir',type=str,default='label_train_2560max/',help='train label')
        parser.add_argument('--tra_edge_dir',type=str,default='edge_train_2560max/',help='train label')

    elif data_name == 'UH':
        # UHRSD
        parser.add_argument('--data_dir', type=str,default='./train_data/UHRSD/', help='Parent folder')
        parser.add_argument('--tra_img_dir',type=str,default='UH_train/img/',help='Location of training image')
        parser.add_argument('--tra_label_dir',type=str,default='UH_train/mask/',help='train label')
        parser.add_argument('--tra_edge_dir',type=str,default='UH_train/edge/',help='train label')

    elif data_name == 'KUH':
        # HRS10K + UHRSD + HRSOD
        parser.add_argument('--data_dir', type=str,default='./train_data/MIX-KUH/', help='Parent folder')
        parser.add_argument('--tra_img_dir',type=str,default='KUH_train/img/',help='Location of training image')
        parser.add_argument('--tra_label_dir',type=str,default='KUH_train/mask/',help='train label')
        parser.add_argument('--tra_edge_dir',type=str,default='KUH_train/edge/',help='train label')

    elif data_name == 'DH':
        # DUTS + HRSOD
        parser.add_argument('--data_dir', type=str,default='./train_data/MIX-DH/', help='Parent folder')
        parser.add_argument('--tra_img_dir',type=str,default='image/',help='Location of training image')
        parser.add_argument('--tra_label_dir',type=str,default='mask/',help='train label')
        parser.add_argument('--tra_edge_dir',type=str,default='edge/',help='train label')

    return parser

# Training params
def training_param(parser):
    parser.add_argument('--init_lr',type=int,default=1e-2,help='learning rate')
    parser.add_argument('--warmup_epoch_num',type=int,default=16,help='warmup epochs')
    parser.add_argument('--train_scheduler_num',type=int,default=32,help='scheduler epochs')
    parser.add_argument('--epoch_num',type=int,default=32,help='Total epochs')
    parser.add_argument('--batchsize',type=int,default=3,help='Batchsize')
    parser.add_argument('--itr_epoch',type=int,default=4980,help='iterations per epoch/4980-KUH/2180-UH/4054-DH/536-HRSOD/2800-HR10K')

    parser.add_argument('--resume',action='store_true', default=False,help='Resume training,False')  
    parser.add_argument('--resume_path',type=str,\
        default='save_models/Atemp/epoch_23.pth',help='resume model path')
    parser.add_argument('--save_interval',type=int,default=1,help='save model every X epoch')


    return parser

def transformer_setting(parser):
    # Transformer setting-----------------------------------------------
    parser.add_argument('--patch_size', type=int,
                        default=4, help='output channel of network')
    parser.add_argument('--in_chans', type=int,
                        default=3, help='output channel of network')                    
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--embed_dim', type=int,
                        default=128, help='Swin setting')                    
    parser.add_argument('--depth', type=tuple,
                        default=[2,2,18,2], help='Swin setting')
    parser.add_argument('--depth_decoder', type=tuple,
                        default=[2,2,2,2], help='Swin setting') 
    parser.add_argument('--num_heads', type=tuple,
                        default=[4,8,16,32], help='Swin setting')  
    parser.add_argument('--window_size', type=int,
                        default=12, help='Swin setting')
    parser.add_argument('--mlp_ratio', type=float,
                        default=4.0, help='Swin setting')                    
    parser.add_argument('--qkv_bias', type=bool,
                        default=True, help='Swin setting')
    parser.add_argument('--qk_scale', type=float,
                        default=None, help='Swin setting')
    parser.add_argument('--drop_rate', type=float,
                        default=0.0, help='Model setting')
    parser.add_argument('--drop_path_rate', type=float,
                        default=0.1, help='Model setting')                    
    parser.add_argument('--ape', type=bool,
                        default=False, help='Swin setting')
    parser.add_argument('--use_pretrain', type=bool,
                        default=True, help='Swin setting')
    parser.add_argument('--patch_norm', type=bool,
                        default=True, help='Swin setting')
    parser.add_argument('--use_checkpoint', type=bool,
                        default=False, help='half precise training')
    parser.add_argument('--pretrained_path', type=str,
                        default='save_models/pretrain/swin_base_patch4_window12_384_22k.pth', 
                        help='model path')
    
    return parser



def myParser():
    my_parser = argparse.ArgumentParser()
    data_name = 'KUH'
    exp_name = 'exp1'

    tensorboard_folder = 'runs1'
    save_path = 'save_models/Atemp/'

    # setting==========================
    # change dataset dir
    my_parser = setting_data(data_name, my_parser)

    # change batchsize, training epoch, iteration num, etc.
    my_parser = training_param(my_parser)

    # if using different backbone, change here
    my_parser = transformer_setting(my_parser)

    my_parser.add_argument('--model_dir',type=str,default=save_path,help='Save directory')
    my_parser.add_argument('--writer_path',type=str,default=tensorboard_folder+'/'+exp_name,help='tensorboard file path') #tensorboard

    args = my_parser.parse_args(args=[])

    return args




if __name__ == "__main__":
    
    cfg = myParser()

    print(cfg)

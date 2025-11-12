from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5,
                                 help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
                                 'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # --- Task model (teacher) ---
        parser.add_argument('--seg_checkpoint', type=str, required=True,
                        help='Path to real-trained segmentation checkpoint (.pth).')
        parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of seg classes in the task model.')
        parser.add_argument('--lambda_seg', type=float, default=1.0,
                        help='Weight for supervised segmentation loss.')
        
        # (optional) task model details
        parser.add_argument('--seg_encoder', type=str, default='vitl')
        parser.add_argument('--seg_decoder', type=str, default='dpt')
        
        # normalization stats used to train the task model:
        # If your checkpoint used ImageNet 0–1 stats, set these to 0.485/0.456/0.406 etc.
        # If it used 0–255 tuples (123.675...), pass those here instead.
        parser.add_argument('--seg_mean', type=float, nargs=3, default=[123.675, 116.28, 103.53])
        parser.add_argument('--seg_std',  type=float, nargs=3, default=[58.395, 57.12, 57.375])
        
        # (optional) center crop fed to the task model; omit if not used
        parser.add_argument('--seg_crop', type=int, nargs=2, default=[504,504])
        
        # (recommended) to keep labels consistent
        parser.add_argument('--ignore_index', type=int, default=255)
            self.isTrain = True

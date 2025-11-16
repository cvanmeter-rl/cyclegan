import torch
import itertools
import torch.nn.functional as F

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# NEW: your frozen task model wrapper
from models.task_model import Task_Network

def labelmap(mask, rules):
    # Initialize mapped_mask with default value of 255, ensuring the same shape as mask
    mapped_mask = np.full(mask.shape, 255, dtype=np.uint8)
    
    # Iterate through each rule in the dictionary
    for src_value, dst_value in rules.items():
        # Ensure src_value is an integer, in case it's not already
        src_value = int(src_value)
        
        # Apply the mapping where mask values match the current rule's source value
        mapped_mask[mask == src_value] = dst_value
    
    return mapped_mask

oem_label_rules = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}


oem_to_dfc19_rules = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 255, 6: 255, 7: 255 }

class CycleGANWithSupervision(BaseModel):
    def name(self):
        return 'CycleGANWithSupervision'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
    
        # ----- losses to log -----
        self.loss_names = [
            'D_A', 'G_A', 'cycle_A', 'idt_A',
            'D_B', 'G_B', 'cycle_B', 'idt_B',
            'seg_AB'   # NEW
        ]

         # ----- visuals to save -----
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
        self.visual_names = visual_names_A + visual_names_B

        # ----- models to save -----
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']
        
        # ----- nets -----
        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
            opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids
        )
        self.netG_B = networks.define_G(
            opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG,
            opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids
        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(
                opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D,
                opt.norm, use_sigmoid, opt.init_type, self.gpu_ids
            )
            self.netD_B = networks.define_D(
                opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D,
                opt.norm, use_sigmoid, opt.init_type, self.gpu_ids
            )

        self.task_net = Task_Network(
            checkpoint_path = opt.seg_checkpoint,     # <- required
            num_classes     = opt.num_classes,        # <- required
            encoder         = getattr(opt, 'seg_encoder', 'vitl'),
            decoder         = getattr(opt, 'seg_decoder', 'dpt'),
            pretrained_backbone = False,
            # Use the SAME stats the task checkpoint was trained with
            norm_mean       = tuple(getattr(opt, 'seg_mean', (123.675, 116.28, 103.53))),
            norm_std        = tuple(getattr(opt, 'seg_std',  (58.395, 57.12, 57.375))),
            input_crop      = getattr(opt, 'seg_crop', None),
            device          = self.device,
        )

        if self.isTrain:
            # pools
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # losses
            self.criterionGAN   = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt   = torch.nn.L1Loss()
            self.criterionSeg   = torch.nn.CrossEntropyLoss(ignore_index=255)  # NEW
            self.lambda_seg     = getattr(opt, 'lambda_seg', 1.0)              # NEW

            # optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # NEW: supervised mask for A→B (synthetic labels)
        self.mask_A = input.get('A_mask', None)
        if self.mask_A is not None:
            self.mask_A = self.mask_A.to(self.device).long()   # [N,H,W]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A  = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B  = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real   = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake   = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D      = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        opt = self.opt
        lambda_idt = opt.lambda_identity
        lambda_A   = opt.lambda_A
        lambda_B   = opt.lambda_B

        # Identity
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # ----- NEW: supervised seg loss on stylized synthetic (A→B) -----
        self.loss_seg_AB = 0.0
        if (self.mask_A is not None) and (self.lambda_seg > 0):
            # Task_Network expects [-1,1]; it internally converts to the task stats
            logits = self.task_net(self.fake_B)      # [N,C,Hs,Ws]
            
            target = self.mask_A                     # [N,H,W], long

            crop_hw = getattr(self.task_net, "input_crop", None)  # e.g., (504,504)
            if crop_hw is not None:
                th, tw = crop_hw
                H, W = target.shape[-2:]
                ys = max((H - th) // 2, 0)
                xs = max((W - tw) // 2, 0)
                target = target[:, ys:ys+th, xs:xs+tw]  # nearest by slicing, preserves ints
            target = target.detach().cpu().numpy().astype(np.uint8)
            target = labelmap(target, oem_label_rules)
            target = labelmap(target,oem_to_dfc19_rules) 
            target = torch.from_numpy(target,astype(np.int64)).to(self.device)
            
            if logits.shape[-2:] != target.shape[-2:]:
                print('logits shape doesnt match')
                target = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=logits.shape[-2:], mode='nearest'
                ).squeeze(1).long()
                print("shape did not match")

            self.loss_seg_AB = self.criterionSeg(logits, target) * self.lambda_seg

        # total G loss
        self.loss_G = (self.loss_G_A + self.loss_G_B +
                       self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_idt_A + self.loss_idt_B +
                       self.loss_seg_AB)
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()


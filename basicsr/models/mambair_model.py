import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from collections import OrderedDict

@MODEL_REGISTRY.register()
class MambaIRModel(SRModel):
    """MambaIR model for image restoration."""

    def optimize_parameters_kd(self, teacher_output, is_mutual=False):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if is_mutual: 
                l_pix = self.cri_pix(self.output, self.gt)
                l_kd = self.cri_pix(self.output, teacher_output)
                loss_dict['l_pix'] = l_pix
                loss_dict['l_kd'] = l_kd
                l_total += 0.5*l_pix
                l_total += 0.5*l_kd
            else: 
                l_pix = self.cri_pix(self.output, self.gt)
                loss_dict['l_pix'] = l_pix
                l_total += l_pix

            
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    # test by partitioning
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class QSSModel(MambaIRModel):
    """MambaIR model for image restoration."""
    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, ssloss = self.net_g(self.lq)
        l_ss = ssloss.mean()
        # import pdb;pdb.set_trace()
        l_total = 0
        l_total += l_ss
        loss_dict = OrderedDict()
        loss_dict['l_ss'] = l_ss
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()
        # import pdb;pdb.set_trace()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out, loss = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class QKDModel(MambaIRModel):
    """MambaIR model for image restoration."""
    
    def optimize_parameters_kd(self, teacher_output, is_mutual=False):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:

            if not is_mutual:
                l_pix = self.cri_pix(self.output, self.gt)
                l_kd = self.cri_pix(self.output, teacher_output)
                loss_dict['l_pix'] = l_pix
                loss_dict['l_kd'] = l_kd
                l_total += 0.5*l_pix
                l_total += 0.5*l_kd
                # l_total += l_kd
            else:
                l_kd1 = self.cri_pix(self.output, teacher_output)
                # l_kd2 = self.cri_pix(teacher_output, self.output)
                loss_dict['l_kd1'] = l_kd1
                # loss_dict['l_kd2'] = l_kd2
                l_total += l_kd1
                # l_total += 0.5*l_kd2
            
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def optimize_parameters_vanilla(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as FF
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

import cv2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            # mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    # width, height = inputs[0][:, :, 0].shape
    height, width = inputs[0][:, :, 0].shape
    img = Image.new('RGB',
                    (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


# progressiveley sampling edge line
def SampleEdgeLineLogits(model, context, mask=None, iterations=1, device='cuda', add_v=0, mul_v=4):
    [img, edge, line] = context
    img = img.to(device)
    edge = edge.to(device)
    line = line.to(device)
    mask = mask.to(device)
    img = img * (1 - mask)
    edge = edge * (1 - mask)
    line = line * (1 - mask)
    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            edge_logits, line_logits = model.forward_with_logits(img.to(torch.float16),
                                                                 edge.to(torch.float16),
                                                                 line.to(torch.float16),
                                                                 masks=mask.to(torch.float16))
            edge_pred = torch.sigmoid(edge_logits)
            line_pred = torch.sigmoid((line_logits + add_v) * mul_v)
            edge = edge + edge_pred * mask
            edge[edge >= 0.25] = 1
            edge[edge < 0.25] = 0
            line = line + line_pred * mask

            b, _, h, w = edge_pred.shape
            edge_pred = edge_pred.reshape(b, -1, 1)
            line_pred = line_pred.reshape(b, -1, 1)
            mask = mask.reshape(b, -1)

            edge_probs = torch.cat([1 - edge_pred, edge_pred], dim=-1)
            line_probs = torch.cat([1 - line_pred, line_pred], dim=-1)
            edge_probs[:, :, 1] += 0.5
            line_probs[:, :, 1] += 0.5
            edge_max_probs = edge_probs.max(dim=-1)[0] + (1 - mask) * (-100)
            line_max_probs = line_probs.max(dim=-1)[0] + (1 - mask) * (-100)

            indices = torch.sort(edge_max_probs + line_max_probs, dim=-1, descending=True)[1]

            for ii in range(b):
                keep = int((i + 1) / iterations * torch.sum(mask[ii, ...]))

                assert torch.sum(mask[ii][indices[ii, :keep]]) == keep, "Error!!!"
                mask[ii][indices[ii, :keep]] = 0

            mask = mask.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
            line = line * (1 - mask)

        return edge, line
    
def SampleEdgeLineLogits_video(model, context, masks=None, iterations=1, device='cuda', add_v=0, mul_v=4):
    [frames, edges, lines] = context
    frames = frames.to(device)
    edges = edges.to(device)
    lines = lines.to(device)
    masks = masks.to(device)

    # Now we assume that the first dimension of img, edge, line, and mask is the batch size
    # and the second dimension is the time dimension.
    # So we need to iterate over these dimensions.
    batch_size, timesteps, _, h, w = frames.shape

    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            edges_logits, lines_logits = model.forward_with_logits(frames, edges, lines, masks=masks)
            
            edges_logits = edges_logits.view(batch_size*timesteps, *edges_logits.shape[2:])
            lines_logits = lines_logits.view(batch_size*timesteps, *lines_logits.shape[2:])
            edges = edges.view(batch_size*timesteps, *edges.shape[2:])
            lines = lines.view(batch_size*timesteps, *lines.shape[2:])
            masks = masks.view(batch_size*timesteps, *masks.shape[2:])
            
            edges_pred = torch.sigmoid(edges_logits)
            lines_pred = torch.sigmoid((lines_logits + add_v) * mul_v)
            edges = edges + edges_pred * masks
            edges[edges >= 0.25] = 1
            edges[edges < 0.25] = 0
            lines = lines + lines_pred * masks

            t, _, h, w = edges_pred.shape
            edges_pred = edges_pred.reshape(t, -1, 1)
            lines_pred = lines_pred.reshape(t, -1, 1)
            masks = masks.reshape(t, -1)

            edges_probs = torch.cat([1 - edges_pred, edges_pred], dim=-1)
            lines_probs = torch.cat([1 - lines_pred, lines_pred], dim=-1)
            edges_probs[:, :, 1] += 0.5
            lines_probs[:, :, 1] += 0.5
            edges_max_probs = edges_probs.max(dim=-1)[0] + (1 - masks) * (-100)
            lines_max_probs = lines_probs.max(dim=-1)[0] + (1 - masks) * (-100)

            indices = torch.sort(edges_max_probs + lines_max_probs, dim=-1, descending=True)[1]

            for ii in range(t):
                keep = int((i + 1) / iterations * torch.sum(masks[ii, ...]))

                assert torch.sum(masks[ii][indices[ii, :keep]]) == keep, "Error!!!"
                masks[ii][indices[ii, :keep]] = 0

            masks = masks.view(batch_size, timesteps, 1, h, w)
            edges = edges.view(batch_size, timesteps, 1, h, w)
            lines = lines.view(batch_size, timesteps, 1, h, w)
            edges = edges * (1 - masks)
            lines = lines * (1 - masks)

    return edges, lines



def get_lr_schedule_with_warmup(optimizer, num_warmup_steps, milestone_step, gamma, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            lr_weight = 1.0
            decay_times = current_step // milestone_step
            for _ in range(decay_times):
                lr_weight *= gamma
        return lr_weight

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def torch_init_model(model, total_dict, key, rank=0):
    state_dict = total_dict[key]
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = FF.to_tensor(img).float()
    return img_t


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]

"""
Below are the tool function reference from FuseFormer
"""
def get_frame_mask_edge_line_list(args):
    if args.dataset == 'davis':
        data_root = "./datasets/DATASET_DAVIS"
        mask_dir = "./datasets/random_mask_stationary_w432_h240"
        frame_dir = os.path.join(data_root, "JPEGImages", "480p")
        edge_dir = os.path.join(data_root, "edges")
        line_dir = os.path.join(data_root, "wireframes")
    elif args.dataset == 'youtubevos':
        data_root = "./datasets/YouTubeVOS/"
        mask_dir = "./datasets/random_mask_stationary_youtube_w432_h240"
        frame_dir = os.path.join(data_root, "test_all_frames", "JPEGImages")
        if args.edge_gaussian == 0:
            edge_dir = os.path.join(data_root, "test_all_frames", "edges_old")
        else :
            edge_dir = os.path.join(data_root, "test_all_frames", "edges")
        line_dir = os.path.join(data_root, "test_all_frames", "wireframes")

    mask_folder = sorted(os.listdir(mask_dir))
    mask_list = [os.path.join(mask_dir, name) for name in mask_folder]
    frame_folder = sorted(os.listdir(frame_dir))
    frame_list = [os.path.join(frame_dir, name) for name in frame_folder]
    edge_folder = sorted(os.listdir(edge_dir))
    edge_list = [os.path.join(edge_dir, name) for name in edge_folder]
    line_folder = sorted(os.listdir(line_dir))
    line_list = [os.path.join(line_dir, name) for name in line_folder]    

    print("[Finish building dataset {}]".format(args.dataset))
    return frame_list, mask_list, edge_list, line_list

# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length, ref_length=5, num_ref=-1):
    ref_index = []
    if num_ref == -1: 
        for i in range(0, length, ref_length): # sample from 0 to length with interval ref_length, try to get the whole story
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
                if len(ref_index) >= num_ref:
                    break
    return ref_index

# read frame-wise masks 
def read_mask(mpath, w, h):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((h, w), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255)) # 0, 255
    return masks

#  read frames from video 
def read_frame_from_videos(vname, w, h):

    lst = os.listdir(vname)
    lst.sort()
    fr_lst = []
    idx_lst = []
    for name in lst:
        fr_lst.append(vname+'/'+name)
        idx_lst.append(name)
    frames = []
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((h,w)))
    return frames, idx_lst  

def read_edge_line_PIL(edge_path, line_path, w, h):
    edgeNames = os.listdir(edge_path)
    edgeNames.sort()
    lineNames = os.listdir(line_path)
    lineNames.sort()

    edge_list, line_list = [], []
    for ename, lname in zip(edgeNames, lineNames):
        edge_list.append(Image.open(os.path.join(edge_path, ename)).convert('L').resize((h,w)))
        line_list.append(Image.open(os.path.join(line_path, lname)).convert('L').resize((h,w)))

    return edge_list, line_list

def create_square_masks(video_length, h, w):
    masks = []
    for i in range(video_length):
        this_mask = np.zeros((h, w))
        this_mask[int(h/4):h-int(h/4), int(w/4):w-int(w/4)] = 1
        this_mask = Image.fromarray((this_mask*255).astype(np.uint8))
        masks.append(this_mask.convert('L'))
    return masks

def get_res_list(dir):
    folders = sorted(os.listdir(dir))
    return [os.path.join(dir, f) for f in folders]

def load_masked_position_encoding(mask, input_size, pos_num, str_size):
        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2] # original size
        ori_mask = ori_mask / 255
        # mask = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, input_size, interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255 # make sure the mask is binary
        h, w = mask.shape[0:2] # resized size
        mask3 = mask.copy() 
        mask3 = 1. - (mask3 / 255.0) # 0 for masked area, 1 for unmasked area
        pos = np.zeros((h, w), dtype=np.int32) # position encoding
        direct = np.zeros((h, w, 4), dtype=np.int32) # direction encoding
        i = 0
        while np.sum(1 - mask3) > 0: # while there is still unmasked area
            i += 1 # i is the index of the current mask
            mask3_ = cv2.filter2D(mask3, -1, np.ones((3, 3), dtype=np.float32)) # dilate the mask
            mask3_[mask3_ > 0] = 1 # make sure the mask is binary
            sub_mask = mask3_ - mask3 # get the newly added area
            pos[sub_mask == 1] = i # set the position encoding

            m = cv2.filter2D(mask3, -1, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32))
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 0] = 1

            m = cv2.filter2D(mask3, -1, np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32))
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 1] = 1

            m = cv2.filter2D(mask3, -1, np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32))
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 2] = 1

            m = cv2.filter2D(mask3, -1, np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32))
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 3] = 1

            mask3 = mask3_

        abs_pos = pos.copy() # absolute position encoding
        rel_pos = pos / (str_size / 2)  # to 0~1 maybe larger than 1
        rel_pos = (rel_pos * pos_num).astype(np.int32) # to 0~pos_num
        rel_pos = np.clip(rel_pos, 0, pos_num - 1) # clip to 0~pos_num-1

        if ori_w != w or ori_h != h: # if the mask is resized
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        return rel_pos, abs_pos, direct
    
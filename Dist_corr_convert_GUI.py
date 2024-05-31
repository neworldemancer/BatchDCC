#!/usr/bin/env python
# coding: utf-8

"""
This GUI interface is the extension & evolution of the BatchSAC GUI.
It aims at easy and batched processign of distortion correction tasks for 2-photon microscopy
The initial version implemented by H. Madi, polished by M. Vladymyrov.
University of Bern, Data Science Lab, 2024
"""
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from utils import imgio as iio
from utils import dist_utils as du
import os
import shutil
import glob

import npy2bdv
import sys 
import re
import wx
import subprocess
import json
import traceback
import nbformat

# from nbconvert.preprocessors import ExecutePreprocessor

scripts_root = r'd:\Trafficking_proc\BatchDCC'

mic_channels_map = {"[2BLUE]": 0, "[5GREEN]": 1, "[6RED]": 2, "[7FarRED]": 3, '[8FarFarRED]': 4}
channels_Symb_map = {"[2BLUE]": 'B', "[5GREEN]": 'G', "[6RED]": 'R', "[7FarRED]": 'FR', '[8FarFarRED]': 'FFR'}


def print_traceback(e):
    t_str_arr = traceback.format_exception(None, e, e.__traceback__)
    t_str = ''.join(t_str_arr)
    print(t_str, file=sys.stderr, flush=True)
    
def get_channels_times(path):
    list_of_files_path = glob.glob(path+'\*.tif')
    list_of_files = [os.path.basename(fp).split('.')[0] for fp in list_of_files_path]
    
    if len(list_of_files) == 0:
        return None
    idx_time = 0
    idx_ch = 0
    idx_chI = 0
    all_parts = list_of_files[0].split(' ')
    for p_idx, part in enumerate(all_parts):
        if '[' in part:
            idx_ch = p_idx
    
        if 'Time' in part:
            idx_time = p_idx

        if '_C' in part:
            idx_chI = p_idx
            
    channels = [f.split(' ')[idx_ch] for f in list_of_files]
    
    channelsI = [f.split(' ')[idx_chI] for f in list_of_files]
    channelsI = [f.split('_')[1] for f in channelsI]
    channelsI = [int(f.split('C')[1]) for f in channelsI]
    
    ch_id_map = {ch_idx:ch_n for ch_n, ch_idx in zip(channels, channelsI)}
    #print(ch_id_map)
    
    #channels = set(channels)
    times = [f.split(' ')[idx_time] for f in list_of_files]
    times = set(times)

    times = list(times)
    times = [int(t.split('Time')[1]) for t in times]
    
    n_t = np.max(times)+1
    n_ch =  np.max(channelsI)+1
    
    return ch_id_map, n_ch, n_t

    

def get_dir(root_dir, pref, doc, pos=None):
    if pos:
        return os.path.join(root_dir, pref+doc+'__pos_'+pos, '')
    else:
        return os.path.join(root_dir, pref+doc, '')


def get_im_name(ds_dir, doc, ch_idx, t_idx, ch_map):
    ch_name = ch_map[ch_idx]
    
    return os.path.join(ds_dir,
                        doc+'_Doc1_PMT - PMT '+ ch_name + ' _C' + '%02d'%ch_idx+ '_Time Time' + '%04d'%t_idx + '.ome.tif'
                       )


def convert_to_h5(ds_dir, doc, out_file_name, res, comp=False):
    ch_map, n_chs, n_times = get_channels_times(ds_dir)

    
    o_tif_fn = out_file_name
    
    try:
        bdv_writer = npy2bdv.BdvWriter(o_tif_fn, nchannels=n_chs, subsamp=((1, 1, 1),)
                                      , compression='gzip' if comp else None
                                      )

        for t_idx in range(n_times):
            for ch_idx in range(n_chs):
                stack = iio.read_mp_tiff(get_im_name(ds_dir, doc, ch_idx, t_idx, ch_map))
                bdv_writer.append_view(stack, time=t_idx, channel=ch_idx, voxel_size_xyz=res, voxel_units='um')

            print(f'{t_idx/n_times*100:.2f} % ', end='\r')

        bdv_writer.write_xml()
        bdv_writer.close()
    except Exception as e:
        print(e)
        print_traceback(e)


class ModalWin(wx.Dialog):
    def __init__(self, parent, dlg_class, pars=None):
        super().__init__(parent=parent, title='DistortionCorrect/Convert 2pm data')
        ####---- Variables
        self.SetEscapeId(12345)
        ####---- Widgets
        self.a = dlg_class(self, pars)
        self.buttonOk = wx.Button(self, wx.ID_OK, label='Next')
        ####---- Sizers
        self.sizerB = wx.StdDialogButtonSizer()
        self.sizerB.AddButton(self.buttonOk)
        
        self.sizerB.Realize()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.a, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(self.sizerB, border=10, 
            flag=wx.ALIGN_RIGHT|wx.ALL)

        self.SetSizerAndFit(self.sizer)

        self.SetPosition(pt=(550, 200))


class PathPanel(wx.Panel):
    def __init__(self, parent, pars):
        super().__init__(parent=parent)
        ####---- Variables
        self.parent = parent
        ####---- Widgets
        label = ("1. Select Dir with datasets")
        self.text = wx.StaticText(self, label=label, pos=(10, 10))
        self.path = wx.TextCtrl(self, value='c:\\', pos=(10, 30))
        
        self.browse_btn = wx.Button(self, -1, "Browse", pos=(160, 30))
        self.Bind(wx.EVT_BUTTON, self.Browse, self.browse_btn)
         
    def Browse(self, event=None):
        try:
            dlg = wx.DirDialog (None, "Choose dataset diretory", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:
                RootPath = dlg.GetPath()
                self.path.SetValue(RootPath)

            dlg.Destroy()
        
        except:
            pass
        
    
class DatasetProcConfigurator(wx.Panel):
    def __init__(self, parent, pars):
        super().__init__(parent=parent)

        ####---- Variables
        self.parent = parent
        self.root_dir, self.runs_info = pars
        
        self.all_ds_t_values = set()  
        for ds_dir, els in self.runs_info.items():
            ds_date, ds_times, ds_ress, ds_nposs = els
            self.all_ds_t_values.update(ds_times)  

        self.all_ds_t_values = sorted(self.all_ds_t_values)  
        
        self.rows = []
        ####---- Widgets
        
        self.sizer = wx.FlexGridSizer(10)
        self.gen_titel()
        
        for ds_dir, els in self.runs_info.items():
            ds_date, ds_times, ds_ress, ds_nposs = els
            
            # find already processed DC dirs:
            processed_dc_dirs = []
            experiment_dir = os.path.join(self.root_dir, ds_dir)
            for s_ds_dir in os.listdir(experiment_dir):
                full_path = os.path.join(experiment_dir, s_ds_dir)
                # Check if the item is a directory and meets the specified condition
                if not os.path.isdir(full_path):
                    continue
                if '_DC__' in s_ds_dir:  # result of application of a corerction do not contain the spline fit required to generate new corr map
                    continue
                name_sections = s_ds_dir.split('_')[-2:]  # expected to end in <...>_DC_<CL>
                if len(name_sections) != 2:  # bad format
                    continue
                if name_sections[0] != 'DC' or name_sections[1] not in ['B', 'G', 'R', 'FR', 'FFR']:
                    continue
                if not os.path.exists(os.path.join(full_path, 'itr_00', 'ia', 'distortion')):
                    continue
                    
                processed_dc_dirs.append(s_ds_dir)
                

            for ds_t, ds_res, ds_npos in zip(ds_times, ds_ress, ds_nposs):
                path = get_dir(f'{self.root_dir}/{ds_dir}', f'{ds_date}_Doc1_', ds_t)
                ch_map, n_chs, n_times = get_channels_times(path)
                self.gen_ds_row(ds_dir, ds_date, ds_t, ds_res, ds_npos, ch_map, processed_dc_dirs)
                
        self.SetSizerAndFit(self.sizer)

    def gen_ds_row(self, ds_dir, ds_date, ds_t, ds_res, ds_npos, ch_map, processed_dc_dirs):
        ds_process = wx.CheckBox(self)
        ds_process.SetValue(True)
        ds_dir_lbl = wx.StaticText(self, label=ds_dir)
        ds_date_lbl = wx.StaticText(self, label=ds_date)
        ds_t_lbl = wx.StaticText(self, label=ds_t)
        dist_phases = wx.TextCtrl(self, value='8')
        dist_samples = wx.TextCtrl(self, value='4')
        ds_res_xy_lbl = wx.TextCtrl(self, value=f'{ds_res[0]:.3f}')
        ds_res_z_lbl = wx.TextCtrl(self, value=f'{ds_res[2]:.3f}')
        ds_align_cb = wx.ComboBox(self, -1, style=wx.CB_READONLY)
        s_dist_dataset = wx.ComboBox(self, -1, style=wx.CB_READONLY)

        channels = sorted(list(ch_map.values()))
        for ch in channels:
            ds_align_cb.Append(ch)
        ds_align_cb.SetSelection(1)  # wx.NOT_FOUND
        channels_len = len(channels) #just added

        dc_files_path = ds_dir
        
        s_dist_dataset.Append("self")
        for item in processed_dc_dirs:
            s_dist_dataset.Append(item)
        s_dist_dataset.SetSelection(0)

        els = {
            'ds_process': ds_process,
            'ds_dir_lbl' : ds_dir_lbl,
            'ds_date_lbl' : ds_date_lbl,
            'ds_t_lbl' : ds_t_lbl,
            'dist_phases' : dist_phases,
            'dist_samples' : dist_samples,
            'ds_res_xy_lbl' : ds_res_xy_lbl,
            'ds_res_z_lbl' : ds_res_z_lbl,
            'ds_align_cb' : ds_align_cb,
            's_dist_dataset' : s_dist_dataset
        }
        
        self.rows.append(els)
        
        self.sizer.Add(ds_process, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_dir_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_date_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_t_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_xy_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_z_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(dist_phases, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(dist_samples, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_align_cb , border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(s_dist_dataset , border=10, flag=wx.ALIGN_LEFT|wx.ALL)

    def gen_titel(self):
        ds_process = wx.StaticText(self, label='process')
        ds_dir_lbl = wx.StaticText(self, label='directory')
        ds_date_lbl = wx.StaticText(self, label='date')
        ds_t_lbl = wx.StaticText(self, label='dataset')
        ds_res_xy_lbl = wx.StaticText(self, label='pixel xy, um')
        ds_res_z_lbl = wx.StaticText(self, label='pixel z, um')
        dist_phases = wx.StaticText(self, label='num dist. phases')
        dist_samples = wx.StaticText(self, label='num dist. samples')
        ds_align_cb = wx.StaticText(self, label='align channel')
        s_dist_dataset = wx.StaticText(self, label='sourse distortion dataset')


        self.sizer.Add(ds_process, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_dir_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_date_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_t_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_xy_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_z_lbl, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(dist_phases, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(dist_samples, border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_align_cb , border=10, flag=wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(s_dist_dataset , border=10, flag=wx.ALIGN_LEFT|wx.ALL)

def get_runs_info(run_root, dirs):
    res = [0.7731239092495636, 0.7731239092495636, 4.0]
    runs_info = {}

    for d in dirs:
        p = os.path.join(run_root, d, '*' )
        dss = glob.glob(p)

        dates = []
        times = []
        ress = []
        n_poss = []
        for ds_i in dss:
            bn = os.path.basename(ds_i)
            if '_bk' in bn or '_ALGN' in bn:
                print('skipping IGNORED dir', bn)
                continue

            if len(glob.glob(os.path.join(ds_i, '*tif'))) == 0:
                print('skipping EMPTY dir', bn)
                continue

            n_pos = 1 if '__pos' in bn else 4

            findres = re.findall('(.*)_Doc1_(.*)', bn)
            if len(findres)==0:
                continue
                
            date, t = findres[0]
            
            
            dates.append(date)
            times.append(t)
            ress.append(res)
            n_poss.append(n_pos)

        dates = list(set(dates))
        assert len(dates)==1

        runs_info[d] = (dates[0], times, ress, n_poss)

    return runs_info


def run_dist_corr_bat(cfg):
    try:
        fname = os.path.join(scripts_root, 'proc_dc.bat')
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.call([fname, f'{cfg}'], startupinfo=si)
    except Exception as e:
        print(e)
        print_traceback(e)
    
def save_res(path, res):
    px_sz_xy, px_sz_z = res
    fn = os.path.join(path, 'resolution.txt')
    
    with open(fn, 'wt') as f:
        f.write(f'{px_sz_xy} {px_sz_z}')
                
def load_res(path):
    fn = os.path.join(path, 'resolution.txt')
    
    with open(fn, 'rt') as f:
        s = f.read()
        res = s.split(' ')
        res = [float(v) for v in res]
        assert 2 <= len(res) <= 3, f'unexpected resolution {res}'
        if len(res) == 3:
                assert np.isclose(res[0], res[1]), f'x, y z pixel sizes provided ({res}) in the file {fn}. The x and y pixel sizes are expected to be same'
                
                res = res[1:]
                
        return res
                
def do_process(i, run_root, run, date, doc, dist_phase, dist_sample,
                           res, process, align_channel, channels, dc_dataset):
    """
    Applies distortion correction to a dataset.
    Distortion is either measured on the current dataset (when `dc_dataset` is not set)
    or the distortion measured on daatset `dc_dataset` is used to apply to this dataset
    """
    
    if not process:
        return
    
    print(f'\n\n Proocessing: run_root={run_root}, run={run}, date={date}, doc={doc}, res={res}')
    
    timestamp = doc[:8]
    
    if dc_dataset == "self":
        cfg_file_path, out_dir = generate_cfg_file_from_ds_list(i, run_root, run, date, doc, 
                                                       dist_phase, dist_sample, res, 
                                                       align_channel, channels)
    else:
        corrected_dataset_dir = dc_dataset  # this is the dataset which is already corrected for distortion, will be smth like "221129_Doc1_11-40-06_DC_FFR"

        # target_dataset_dir = dc_dataset 
        target_dataset_dir = f'{date}_Doc1_{doc}'  # this is the dataset to which correction will be applied.
        experiments_path = os.path.join(run_root, run)
        target_px_sz_xy, target_px_sz_z = res
        
        corrected_dataset_path = os.path.join(experiments_path, corrected_dataset_dir) #change in the second cfg
        corr_px_sz_xy, corr_px_sz_z = load_res(corrected_dataset_path)

        # ensure that the dataset xy resolution for the target dataset is same as the original dataset
        if not np.isclose(corr_px_sz_xy, target_px_sz_xy).all():
            raise ValueError('The dataset xy resolution for the target dataser is not same as the original dataset')
                
        #do process
        
        #checks to perform
        # ensure that the dataset is corrected for distortion - contains the "_DC" in directory name but not "_DC__"
        if not '_DC' in corrected_dataset_dir and '_DC__' not in corrected_dataset_dir:
            raise ValueError('The dataset is not corrected for distortion')

        #preprocessing 2
        corrected_target_dataset_dir = target_dataset_dir + '_DC__' + corrected_dataset_dir 
        corrected_target_dataset_path = os.path.join(experiments_path, corrected_target_dataset_dir)
        corrected_target_dataset_dist_path = os.path.join(corrected_target_dataset_path, 'itr_00', 'ia', 'distortion') #ExistingDistMapDir in cfg
        os.makedirs(corrected_target_dataset_dist_path, exist_ok=True)

        pxsz = (corr_px_sz_xy, -corr_px_sz_xy, -corr_px_sz_z)

        z_scale_factor = corr_px_sz_z / target_px_sz_z
        # get the size in pixels of the target dataset
        target_dataset_path = os.path.join(experiments_path, target_dataset_dir)

        # get the first tiff file in the target dataset
        tiff_files = glob.glob(target_dataset_path + '\\*.ome.tif')
        if len(tiff_files) == 0:
            raise ValueError('No tiff files found in the target dataset directory')

        # get the size of the first tiff file
        image_file = tiff_files[0]

        img = iio.read_mp_tiff(image_file)
        target_sz = img.shape[::-1]
        
        # save a plot given the experiment path, dataset dir, and pixel size 
        dist_path = os.path.join(corrected_dataset_path, 'dist_info')
        os.makedirs(dist_path, exist_ok=True)

        plot_save_path = os.path.join(dist_path, 'plot')
        labels = ['dist curve']
        
        du.compare_px([corrected_dataset_path], labels, comp_name=plot_save_path, pxsz=pxsz, fig_size=(16, 5), npix=None, show=False)
        
        res_gsdm= du.gen_save_dist_map(path=corrected_dataset_path,   # path to the previously corrected dataset
                                        out_path=corrected_target_dataset_dist_path,  # path to save the generated distortion map 
                                        pxsz=(1, 1, z_scale_factor),
                                        orig_sz=target_sz)
        if not res_gsdm:
            return  # can't proceed
        
        cfg_file_path, out_dir  = generate_apply_cfg_file(i, run_root, run, date, doc, 
                                                   res, channels,
                                                   corrected_target_dataset_path, corrected_target_dataset_dist_path
                                               )
        
    run_dist_corr_bat(cfg_file_path)
    out_file_name_h5 = os.path.dirname(out_dir)+'.h5'
    convert_to_h5(out_dir, doc, out_file_name_h5, res=(res[0], res[0], res[1]), comp=True)


def process_all(ds_list):
    for i, (run_root, run, date, doc, dist_phase, dist_sample,
            res, process, align_channel, channels, dc_dataset) in enumerate(ds_list):
        if process:
            try:
                do_process(i, run_root, run, date, doc, dist_phase, dist_sample,
                           res, process, align_channel, channels, dc_dataset)
            except Exception as e:
                print(e)
                print_traceback(e)

                

def generate_cfg_file_from_ds_list(run_idx, run_root, run, date, doc, 
                                   dist_phase, dist_sample, res, 
                                   align_channel, channels):
    # Define the name of  template file
    template_path = os.path.join(scripts_root, 'cfg_templates', 'DistCorr_do_dc_corrected.cfg')
    experiments_path = os.path.join(run_root, run)

    n_channels = len(channels)

    ref_channel = -1
    for j, ch in enumerate(channels):
        if ch == align_channel:
            ref_channel = j
            break
    assert ref_channel != -1, 'unexpected error: align channel not in lost of channels'
    
    mic_channels = [mic_channels_map[channel] for channel in channels]
    ref_channel_symb = channels_Symb_map[align_channel]
    
    with open(template_path, 'r') as file:
        cfg_template = json.load(file)

    # Update the template with your variables,
    corr_dir = f'{date}_Doc1_{doc}_DC_{ref_channel_symb}'
    
    output_path = os.path.join(experiments_path, corr_dir)  # should be "221129_Doc1_14-38-44_DC_<CL>"
    os.makedirs(output_path, exist_ok=True)
    save_res(output_path, res)
                
    cfg_template["DistCorr"]["OutputPath"] =  output_path
    cfg_template["DistCorr"]["NChannel"] = f'{n_channels}'
    cfg_template["DistCorr"]["RefChannel"] = f'{ref_channel}'
    cfg_template["DistCorr"]["InputPath"] = os.path.join(experiments_path, f'{date}_Doc1_{doc}')  # ? doc -> f'{date}_Doc1_{doc}'
    for j in range(n_channels):
         cfg_template["DistCorr"][f"Channel_{j}"] = f'{doc}_Doc1_PMT - PMT {channels[j]} _C{mic_channels[j]:02d}_Time Time%04d.ome.tif'
    cfg_template["DistCorr"]["NDistPhasese"] = dist_phase
    cfg_template["DistCorr"]["NDistSamples"] = dist_sample
    # Update other necessary keys similarly

    cfg_file_name = f"dist_corr_config.cfg"  # Use i for unique naming
    cfg_file_path = os.path.join(output_path, cfg_file_name)

    # Corrected variable name here
    with open(cfg_file_path, 'w') as cfg_file:
        json.dump(cfg_template, cfg_file, indent=4)

    return cfg_file_path, os.path.join(output_path, 'corr_full')

        
def generate_apply_cfg_file(run_idx, run_root, run, date, doc, 
                            res, channels, tgt_dir, dist_map_dir):
    
    # Define the name of your template file
    template_path = os.path.join(scripts_root, 'cfg_templates', 'DistCorr_apply_dc.cfg')
    experiments_path = os.path.join(run_root, run)
    
    n_channels = len(channels)
    
    mic_channels = [mic_channels_map[channel] for channel in channels]
    #ref_channel_symb = channels_Symb_map[align_channel]


    
    with open(template_path, 'r') as file:
        cfg_template = json.load(file)

    # Update the template with your variables,
    
    output_path = tgt_dir
    save_res(output_path, res)
                
    cfg_template["DistCorr"]["OutputPath"] =  output_path
    cfg_template["DistCorr"]["NChannel"] = f'{n_channels}'
    cfg_template["DistCorr"]["InputPath"] = os.path.join(experiments_path, f'{date}_Doc1_{doc}')  # ? doc -> f'{date}_Doc1_{doc}'
    
    for j in range(n_channels):
         cfg_template["DistCorr"][f"Channel_{j}"] = f'{doc}_Doc1_PMT - PMT {channels[j]} _C{mic_channels[j]:02d}_Time Time%04d.ome.tif'
    cfg_template["DistCorr"]["ExistingDistMapDir"] = dist_map_dir

    # Update other necessary keys similarly

    cfg_file_name = f"dist_corr_config.cfg"  # Use i for unique naming
    cfg_file_path = os.path.join(output_path, cfg_file_name)

    # Corrected variable name here
    with open(cfg_file_path, 'w') as cfg_file:
        json.dump(cfg_template, cfg_file, indent=4)

    return cfg_file_path, os.path.join(output_path, 'corr_full')
        
        
def run():
    app = wx.App()
    cont = True

    if cont:
        frameM = ModalWin(None, PathPanel)
        if frameM.ShowModal() == wx.ID_OK:
            #print("Exited by Ok button")
            pass
        else:
            #print("Exited by X button")
            cont = False
            
        path = frameM.a.path.Value
        frameM.Destroy()

    if cont:

        run_root = os.path.dirname(path)

        dirs = [os.path.basename(path)]
        runs_info = get_runs_info(run_root, dirs)
        
        frameM = ModalWin(None, DatasetProcConfigurator, (run_root, runs_info))
        if frameM.ShowModal() == wx.ID_OK:
            #print("Exited by Ok button")
            pass
        else:
            #print("Exited by X button")
            cont = False
            
        ds_list = []
        for row in frameM.a.rows:
            process = row['ds_process'].Value
            run = row['ds_dir_lbl'].Label
            date = row['ds_date_lbl'].Label
            doc  = row['ds_t_lbl'].Label
            rs_xy = float(row['ds_res_xy_lbl'].Value)
            rs_z = float(row['ds_res_z_lbl'].Value)
            align_channel = row['ds_align_cb'].Value
            dist_phase= row['dist_phases'].GetValue()
            dist_sample= row['dist_samples'].GetValue()
            channels = row['ds_align_cb'].GetItems()
            dc_dataset = row['s_dist_dataset'].GetValue()
                
            ds_list.append([run_root, run, date, doc, dist_phase, dist_sample,
                            [rs_xy, rs_z], process, align_channel, channels, dc_dataset])

        frameM.Destroy()
    app.MainLoop()    
    
    if cont:
        process_all(ds_list)
    
    print('\n\n\n DONE!')

if __name__=='__main__':
    run()

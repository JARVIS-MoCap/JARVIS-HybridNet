import os,sys,inspect
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from lib.vortex.modules.v2vnet.repro_layer import ReprojectionLayer
from lib.dataset.dataset3D import VortexDataset3D
from lib.vortex.modules.efficienttrack.model import EfficientTrackBackbone
import lib.vortex.modules.efficienttrack.darkpose as darkpose
from lib.vortex.utils import ReprojectionTool
from lib.vortex.modules.v2vnet.model import V2VNet

#        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#        self.starter.record()
#        self.ender.record()


class VortexBackbone(nn.Module):
    def __init__(self, cfg, intrinsic_paths, extrinsic_paths, img_size, lookup_path = None):
        super(VortexBackbone, self).__init__()
        self.cfg = cfg
        self.root_dir = cfg.DATASET.DATASET_ROOT_DIR
        self.register_buffer('grid_size', torch.tensor(cfg.VORTEX.ROI_CUBE_SIZE))
        self.register_buffer('grid_spacing', torch.tensor(cfg.VORTEX.GRID_SPACING))
        self.register_buffer('img_size', torch.tensor(img_size))

        self.effTrack = EfficientTrackBackbone(self.cfg, compound_coef=self.cfg.EFFICIENTTRACK.COMPOUND_COEF)
        #self.effTrack.load_state_dict(torch.load('/home/trackingsetup/Documents/Vortex/projects/handPose_test/efficienttrack/models/Colleen_d2_Run2/EfficientTrack-d3_80_171153.pth'), strict = True)
        self.effTrack.requires_grad_(False)
        #self.effTrack.backbone_net.requires_grad_(False)


        self.reproLayer = ReprojectionLayer(cfg, intrinsic_paths, extrinsic_paths, lookup_path)
        self.v2vNet = V2VNet(cfg.EFFICIENTTRACK.NUM_JOINTS, cfg.EFFICIENTTRACK.NUM_JOINTS)
        self.softplus = nn.Softplus()
        self.xx,self.yy,self.zz = torch.meshgrid(torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda(),
                                                 torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda(),
                                                 torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda())
        self.last_time = 0
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)



    def forward(self, imgs, centerHM, center3D):
        batch_size = imgs.shape[0]
        heatmaps_batch =  self.effTrack(imgs.reshape(-1,imgs.shape[2], imgs.shape[3], imgs.shape[4]))[1]
        heatmaps_batch = heatmaps_batch.reshape(batch_size, -1, heatmaps_batch.shape[1], heatmaps_batch.shape[2], heatmaps_batch.shape[3])

        heatmaps_padded = torch.cuda.FloatTensor(imgs.shape[0], imgs.shape[1], heatmaps_batch.shape[2], self.img_size[1], self.img_size[0])
        heatmaps_padded.fill_(0)
        for i in range(imgs.shape[1]):
            heatmaps = heatmaps_batch[:,i]
            for batch, heatmap in enumerate(heatmaps):
                heatmaps_padded[batch,i] = F.pad(input=heatmap,
                                                 pad=(((centerHM[batch,i,0]/2)-heatmap.shape[-1]/2).int(),
                                                      self.img_size[0]-((centerHM[batch,i,0]/2)+heatmap.shape[-1]/2).int(),
                                                      ((centerHM[batch,i,1]/2)-heatmap.shape[-1]/2).int(),
                                                      self.img_size[1]-((centerHM[batch,i,1]/2)+heatmap.shape[-1]/2).int()),
                                                 mode='constant', value=0)

        heatmaps3D = self.reproLayer(heatmaps_padded, center3D)
        heatmap_final = self.v2vNet(((heatmaps3D/255.)))
        heatmap_final = self.softplus(heatmap_final)

        norm = torch.sum(heatmap_final, dim = [2,3,4])
        x = torch.mul(heatmap_final, self.xx)
        x = torch.sum(x, dim = [2,3,4])/norm
        y = torch.mul(heatmap_final, self.yy)
        y = torch.sum(y, dim = [2,3,4])/norm
        z = torch.mul(heatmap_final, self.zz)
        z = torch.sum(z, dim = [2,3,4])/norm
        points3D = torch.stack([x,y,z], dim = 2)
        points3D = (points3D*self.grid_spacing*2-self.grid_size/self.grid_spacing+center3D)
        #torch.cuda.synchronize()
        #self.last_time = self.v2vNet.starter.elapsed_time(self.v2vNet.ender)
        return heatmap_final, heatmaps_padded, points3D


if __name__ == "__main__":
    from config import cfg
    cfg.merge_from_file('/home/timo/Desktop/VoRTEx/projects/Test/config.yaml')

    import time
    training_set = VortexDataset3D(cfg = cfg, set='val')
    vortex = VortexBackbone(cfg, training_set.coco.dataset['calibration']['intrinsics'], training_set.coco.dataset['calibration']['extrinsics'], [640,512], '/home/timo/Desktop/VoRTEx/lookup.npy').cuda()


    #vortex.load_state_dict(torch.load('/home/trackingsetup/Documents/Vortex/projects/handPose_test/vortex/models/Colleen_d3_v2v_ks3_2/Vortex-d_5.pth'), strict = False)
    vortex.requires_grad_(False)
    vortex.eval()
    vortex = vortex.cuda()
    tot_errors_hm3d = []
    tot_errors_class = []
    joint_lengths_hm3d = [[] for i in range(15)]
    joint_lengths_class = [[] for i in range(15)]

    pointsNet = []
    points2DNet = []
    pointsGT = []

    print (len(training_set.image_ids))

    for item in range(len(training_set.image_ids)):
        image_info = training_set.coco.loadImgs(training_set.image_ids[item])[0]
        sample = training_set.__getitem__(item)
        imgs = torch.from_numpy(np.array(sample[0])).cuda()
        imgs_p = torch.unsqueeze(imgs.permute(0,3,1,2).float(),0)
        centerHM = torch.cuda.IntTensor(sample[2])
        center3D = torch.cuda.FloatTensor(sample[3])
        #print (center3D.shape)
        #input = (imgs_p,torch.unsqueeze(centerHM,0),  torch.unsqueeze(center3D,0))
        #import onnx

        #torch.onnx.export(voxelNet, input, 'test.onnx', input_names=['input'],
        #              output_names=['output'], export_params=True, opset_version=11, do_constant_folding = True, use_external_data_format=True)

        with torch.cuda.amp.autocast():
            heatmap3D, heatmaps_padded, points3D_net = vortex(imgs_p,torch.unsqueeze(centerHM,0),  torch.unsqueeze(center3D,0))
            start_time = time.time()
            heatmap3D, heatmaps_padded, points3D_net = vortex(imgs_p,torch.unsqueeze(centerHM,0),  torch.unsqueeze(center3D,0))
        print (vortex.last_time)
        print ((time.time()-start_time)*1000)
        preds, maxvals = darkpose.get_final_preds(heatmaps_padded[0].clamp(0,255).cpu().numpy(), None)
        preds *= 2
        reproTool = ReprojectionTool('T', training_set.root_dir, training_set.coco.dataset['calibration']['intrinsics'], training_set.coco.dataset['calibration']['extrinsics'])

        points3D_rec = []
        for i in range(len(preds[0])):
            point = reproTool.reconstructPoint(np.transpose(preds[:,i,:]))
            points3D_rec.append(point)


        #img = (v[2][0].numpy()*training_set.cfg.DATASET.STD+training_set.cfg.DATASET.MEAN)
        #img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
        #colored_heatmap = cv2.applyColorMap(heatmaps_padded[0,0].cpu().byte().numpy(), cv2.COLORMAP_JET)
        #img = cv2.resize(img*255, (heatmaps_padded[0,0].shape[1], heatmaps_padded[0,0].shape[0])).astype(np.uint8)
        #img = cv2.addWeighted(img,1.0,colored_heatmap,0.4,0)
        #number_of_lines= 23
        #cm_subsection = np.linspace(0.0, 1.0, number_of_lines)
        #colors = [(cm.jet(x)[0]*255,cm.jet(x)[1]*255,cm.jet(x)[2]*255) for x in cm_subsection ]
        #for i, (x, y) in enumerate(preds[0]):
        #    img = cv2.circle(img, (int(np.round(x/2)), int(np.round(y/2))), 3, colors[22-i], 4)

        #cv2.imshow('', img)
        #cv2.waitKey(0)
        heatmap3D = heatmap3D.cpu().numpy()[0]
        center3D = center3D.cpu().numpy()
        if center3D[2] > 1000:
            print (image_info['file_name'])



        figure = plt.figure()
        axes = figure.gca(projection='3d')

        points3D_hm = []

        #xx,yy,zz = torch.meshgrid(torch.arange(40), torch.arange(40), torch.arange(40))

        #xx,yy,zz = torch.meshgrid(torch.arange(52*2), torch.arange(52*2), torch.arange(52*2))
        #for i,heatmap in enumerate(heatmap3D):
        #    heatmap = torch.from_numpy(heatmap)
        #    heatmap[heatmap < 50 ] = 0
        #    norm = torch.sum(heatmap)
        #    x = torch.mul(heatmap, (xx))
        #    x = torch.sum(x)/norm
        #    y = torch.mul(heatmap, (yy))
        #    y = torch.sum(y)/norm
        #    z = torch.mul(heatmap, (zz))
        #    z = torch.sum(z)/norm
        #    points3D_hm.append([x*2.+center3D[0]-104.,y*2.+center3D[1]-104.,z*2.+center3D[2]-104.])



        colors = [(0,0,1,max(0,3./2*c-0.5)) for c in np.linspace(0,1,100)]
        cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
        xx,yy,zz = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing='ij')
        #axes.scatter(xx*grid_spacing+center3D[0]-100,yy*grid_spacing+center3D[1]-100,zz*grid_spacing+center3D[2]-100, c = heatmap3D[0,xx,yy,zz], cmap = cmapblue)

        c = ['r', 'r','r','r','b','b','b','b','g','g','g','g', 'orange', 'orange','orange','orange', 'y','y','y','y','purple', 'purple','purple']
        line_idxs = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [8,9], [9,10], [10,11], [12,13], [13,14], [14,15], [16,17], [17,18], [18,19], [3,7], [7,11], [11,15], [3,21], [7,21],[11,22], [15,22],[21,22], [18,15], [19,22]]
        #for i, point in enumerate(points3D_hm):
        #    print ("HM:", i, points3D_hm[i])
        #    axes.scatter(point[0], point[1], point[2], color = c[i])
        #for line in line_idxs:
        #    axes.plot([points3D_hm[line[0]][0], points3D_hm[line[1]][0]], [points3D_hm[line[0]][1], points3D_hm[line[1]][1]], [points3D_hm[line[0]][2], points3D_hm[line[1]][2]], c = 'gray')
        #if item == 0:
        points3D_net = points3D_net[0].cpu().detach().numpy()
        for i, point in enumerate(points3D_net):
            #print ("HM:", i, points3D_hm[i])
            if i != 20:
                axes.scatter(point[0], point[1], point[2], color = c[i])
        for line in line_idxs:
            axes.plot([points3D_net[line[0]][0], points3D_net[line[1]][0]], [points3D_net[line[0]][1], points3D_net[line[1]][1]], [points3D_net[line[0]][2], points3D_net[line[1]][2]], c = 'gray')

        keypoins3D = sample[1]
        #for idx in line_idxs:
        #    dist = np.linalg.norm(keypoins3D[idx[0]]-keypoins3D[idx[1]])
        #    if dist > 1:
        #        print (item)

        #figure = plt.figure()
        #axes = figure.gca(projection='3d')
        for i, point in enumerate(keypoins3D):
            #print ("Classic:", i, point)
            if i != 20:
                axes.scatter(point[0], point[1], point[2], color = "gray")
        for line in line_idxs:
            axes.plot([keypoins3D[line[0]][0], keypoins3D[line[1]][0]], [keypoins3D[line[0]][1], keypoins3D[line[1]][1]], [keypoins3D[line[0]][2], keypoins3D[line[1]][2]], c = 'gray')

        axes.set_xlim3d(center3D[0]-100, center3D[0]+100)
        axes.set_ylim3d(center3D[1]-100, center3D[1]+100)
        axes.set_zlim3d(center3D[2]-100, center3D[2]+100)
        plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
        #plt.close()

        #plt.show()

        #points3D_net = points3D_net.cpu().numpy()

        #for i in range(23):
            #print (np.linalg.norm(np.array(points3D_rec[i])-keypoins3D[i]))
        #    errors.append(np.linalg.norm(np.array(points3D_hm[i])-keypoins3D[i]))
        #print ('3DHM:', np.mean(np.array(errors)))

        for i,idx in enumerate(line_idxs[0:15]):
            if np.linalg.norm(np.array(points3D_net[idx[0]]-points3D_net[idx[1]])) > 100:
                print ("WHZZZZZZZZZZZ", np.linalg.norm(np.array(points3D_net[idx[0]]-points3D_net[idx[1]])))
            joint_lengths_hm3d[i].append(np.linalg.norm(np.array(points3D_net[idx[0]]-points3D_net[idx[1]])))
            #print ("3D", np.linalg.norm(np.array(points3D_net[idx[0]]-points3D_net[idx[1]])))
            #joint_lengths_class[i].append(np.linalg.norm(np.array(points3D_rec[idx[0]]-points3D_rec[idx[1]])))
            joint_lengths_class[i].append(np.linalg.norm(np.array(keypoins3D[idx[0]]-keypoins3D[idx[1]])))

            #print ("STD", np.linalg.norm(np.array(points3D_rec[idx[0]]-points3D_rec[idx[1]])))

        pointsNet.append(points3D_net)
        points2DNet.append(points3D_rec)
        pointsGT.append(keypoins3D)

        errors_3dHM = []
        errors_Std = []
        for i in range(23):
            if i != 20:
                #print (np.linalg.norm(np.array(points3D_rec[i])-keypoins3D[i]))
                errors_3dHM.append(np.linalg.norm(np.array(points3D_net[i])-keypoins3D[i]))
                errors_Std.append(np.linalg.norm(np.array(points3D_rec[i])-keypoins3D[i]))
        print ('3DHM:', np.mean(np.array(errors_3dHM)))
        print ('Std:', np.mean(np.array(errors_Std)))
        if (np.mean(np.array(errors_3dHM)) > 30):
            plt.show()
        else:
            plt.show()

        tot_errors_hm3d.append(np.mean(np.array(errors_3dHM)))
        tot_errors_class.append(np.mean(np.array(errors_Std)))

    print ("Var HM3D:", np.mean(np.var(np.array(joint_lengths_hm3d), axis = 1)))
    print ("Var Class:", np.mean(np.var(np.array(joint_lengths_class), axis = 1)))
    print ("Total 3DHM:", np.mean(tot_errors_hm3d))
    print ("Total Classic:", np.mean(tot_errors_class))

    from numpy import savetxt
    savetxt('points3DNet.csv', np.array(pointsNet).reshape((-1, 23*3)), delimiter=',')
    savetxt('points2DNet.csv', np.array(points2DNet).reshape((-1, 23*3)), delimiter=',')
    savetxt('pointsGT.csv', np.array(pointsGT).reshape((-1, 23*3)), delimiter=',')

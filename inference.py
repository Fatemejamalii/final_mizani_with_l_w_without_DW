from pathlib import Path

from tqdm import tqdm
import tensorflow as tf
from model import landmarks
from writer import Writer
from utils import general_utils as utils
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib.image import imread 


class Inference(object):
    def __init__(self, args, model):
        self.args = args
        self.G = model.G
        self.pixel_loss_func = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)

    def opt_infer_pairs(self):
        names = [f for f in self.args.id_dir.iterdir() if f.suffix[1:] in self.args.img_suffixes]
        # names.extend([f for f in self.args.attr_dir.iterdir() if f.suffix[1:] in self.args.img_suffixes])
        for img_name in tqdm(names):
            id_path = utils.find_file_by_str(self.args.id_dir, img_name.stem)
            mask_path = utils.find_file_by_str(self.args.mask_dir, img_name.stem)
            attr_path = utils.find_file_by_str(self.args.attr_dir, img_name.stem)
            if len(id_path) != 1 or len(attr_path) != 1:
                print(f'Could not find a single pair with name: {img_name.stem}')
                continue

            id_img = utils.read_image(id_path[0], self.args.resolution)
            gt_img = id_img
            mask_img , attr_img = utils.read_mask_image(id_path[0], mask_path[0], self.args.resolution)
		
            out_img = self.G(mask_img, attr_img, id_img)[0]
            id_embedding = self.G(mask_img, attr_img, id_img)[1]
            attr_embedding = self.G(mask_img, attr_img, id_img)[2]			
            z_tag = tf.concat([id_embedding, attr_embedding], -1)
            w = self.G.latent_spaces_mapping(z_tag)
            pred = self.G.stylegan_s(w)
            pred = (pred + 1) / 2
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1 =0.9, beta_2=0.999, epsilon=1e-8 ,name='Adam')
            loss =  tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
            mask = Image.open(mask_path[0])
            mask = mask.convert('RGB')
            mask = mask.resize((256,256))
            mask = np.asarray(mask).astype(float)/255.0
            mask1 = np.asarray(mask).astype(float)  
            # mask1 = np.expand_dims(mask1 , axis=0) 
            loss_value = 0
            wp = tf.Variable(w ,trainable=True)
            for i in range(5000):
                print('iteration:{0}   loss value is: {1}'.format(i,loss_value))
                with tf.GradientTape() as tape:
                    out_img = self.G.stylegan_s(wp) 
                    out_img = (out_img + 1)  / 2 
                    # utils.save_image(out_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_out.png'))                        
                    mask_out_img = out_img * mask1
                    if i%200==0:
                        utils.save_image(out_img , self.args.output_dir.joinpath(f'{img_name.name[:-4]}' + '_out_{0}.png'.format(i)))
                    # utils.save_image(mask_out_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_test.png'))
                    # utils.save_image(mask_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_m.png'))
                    loss_value = loss(mask_img ,mask_out_img)
                                        
                grads = tape.gradient(loss_value, [wp])
                optimizer.apply_gradients(zip(grads, [wp]))
                
            
            opt_pred = self.G.stylegan_s(wp)
            opt_pred = (opt_pred + 1) / 2

            utils.save_image(pred, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_init.png'))
            utils.save_image(opt_pred, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_final.png'))
            utils.save_image(id_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_gt.png'))


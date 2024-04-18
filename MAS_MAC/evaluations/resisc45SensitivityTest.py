import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import os
import csv
import argparse
import numpy as np
import time
from captum.attr import  GuidedBackprop

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.attribution_methods import saliencyMethods as attribution
from util.test_methods import MASTestFunctions as MAS
from util.test_methods import PICTestFunctions as PIC_Test
from util.modified_models import resnet, vit

model = None

# standard resisc45 normalization
transform_normalize = transforms.Normalize(
    mean=[0.368, 0.381, 0.344],
    std=[0.200, 0.181, 0.181]
)

def get_variations_constant(attribution):
    five_perc = 0.05 * attribution.max() 
    ten_perc = 0.10 * attribution.max() 
    twenty_five_perc = 0.25 * attribution.max() 
    fifty_perc = 0.50 * attribution.max() 
    attr_base = np.abs(np.sum(attribution, axis = 2))

    attr_add_5  = (attr_base + five_perc)
    attr_add_10  = (attr_base + ten_perc)
    attr_add_25  = (attr_base + twenty_five_perc)
    attr_add_50 = (attr_base + fifty_perc)

    return [attr_base, attr_add_5, attr_add_10, attr_add_25, attr_add_50]

def get_variations_uniform(attribution):
    five_perc = 0.005 * np.abs(np.sum(attribution, axis = 2)).max() 
    ten_perc = 0.01 * np.abs(np.sum(attribution, axis = 2)).max() 
    twenty_five_perc = 0.02 * np.abs(np.sum(attribution, axis = 2)).max() 
    fifty_perc = 0.03 * np.abs(np.sum(attribution, axis = 2)).max() 

    attr_add_5  = np.abs(np.sum(attribution + np.random.uniform(0, five_perc, attribution.shape), axis = 2))
    attr_add_10  = np.abs(np.sum(attribution + np.random.uniform(0, ten_perc, attribution.shape), axis = 2))
    attr_add_25  = np.abs(np.sum(attribution + np.random.uniform(0, twenty_five_perc, attribution.shape), axis = 2))
    attr_add_50 = np.abs(np.sum(attribution + np.random.uniform(0, fifty_perc, attribution.shape), axis = 2))
    attribution = np.abs(np.sum(attribution, axis = 2))

    return [attribution, attr_add_5, attr_add_10, attr_add_25, attr_add_50]

def run_and_save_tests(noise_type, img_hw, transform, random_mask, saliency_thresholds, image_count, function_steps, batch_size, model, modified_model, model_name, device, dataset_path, dataset_name):
    # initialize RISE and MAS blur kernel
    klen = 11
    ksig = 11
    kern = MAS.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

    MAS_insertion = MAS.MASMetric(model, img_hw * img_hw, 'ins', img_hw, substrate_fn = blur)
    MAS_deletion = MAS.MASMetric(model, img_hw * img_hw, 'del', img_hw, substrate_fn = torch.zeros_like)

    if dataset_name == "resisc45":
        num_classes = 45

    images_per_class = int(np.ceil(image_count / num_classes))
    classes_used = [0] * num_classes

    # the SIC and AIC tests return curves with 1001 values
    SIC_curves = np.zeros((5, 1001))
    AIC_curves = np.zeros((5, 1001))

    # the RISE and MAS tests return curves with img_hw + 1 values
    RISE_ins_curves = np.zeros((5, img_hw + 1))
    RISE_del_curves = np.zeros((5, img_hw + 1))
    MAS_ins_curves = np.zeros((5, img_hw + 1))
    MAS_del_curves = np.zeros((5, img_hw + 1))

    data = datasets.ImageFolder(root = dataset_path, transform=transform)
    data_len = len(data)
    train_split = int(data_len * 0.7)
    val_split = data_len - train_split
    _, val_set = torch.utils.data.random_split(data, [train_split, val_split], torch.Generator().manual_seed(42))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

    counter = 0

    for i, (img, target_class) in enumerate(val_loader):
        if counter == image_count:
            print("method finished")
            break

        img = img.squeeze()
        # only rgb images can be classified
        if img.shape != (3, img_hw, img_hw):
            continue

        tensor_img = transform_normalize(img)
        tensor_img = torch.unsqueeze(tensor_img, 0)

        # tensor_img is in form needed for prediction for the ins/del method
        ins_del_img = tensor_img
        # image for AIC/SIC test
        AIC_SIC_img = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))

        # check if classification is correct
        model_pred = model_utils.getClass(tensor_img, model, device)
        if model_pred.cpu() != target_class:
            continue
    
        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       

        start = time.time()

        ########  Gradient  ########
        gradient = model_utils.getGradients(tensor_img, model, device, target_class)
        grad = np.transpose(gradient.squeeze().detach().cpu().numpy(), (1, 2, 0))

        ########  LIG  ########
        saliency_map = attribution.IG(tensor_img, model, function_steps, batch_size, .9, 0, device, target_class)
        lig = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))

        ########  Guided Backprop  ########
        guided_bp = GuidedBackprop(modified_model)
        tensor_img.requires_grad = True
        saliency_map = guided_bp.attribute(tensor_img.to(device), target = target_class)
        tensor_img.requires_grad = False
        gbp = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))

        # use abs val of attribution map pixels for testing
        grad_test = np.abs(np.sum(grad, axis = 2))
        lig_test = np.abs(np.sum(lig, axis = 2))
        gbp_test = np.abs(np.sum(gbp, axis = 2))

        if np.sum(lig_test.reshape(1, 1, img_hw **2)) == 0 or np.sum(grad_test.reshape(1, 1, img_hw **2)) == 0 or np.sum(gbp_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 ORIG attribution")
            classes_used[target_class] -= 1
            continue

        if noise_type == "constant":
            grad_variations = get_variations_constant(grad)
            LIG_variations = get_variations_constant(lig)
            GBP_variations = get_variations_constant(gbp)
        elif noise_type == "noised":
            grad_variations = get_variations_uniform(grad)
            LIG_variations = get_variations_uniform(lig)
            GBP_variations = get_variations_uniform(gbp)

        print(model_name + " " + noise_type + " sensitivity test, " + dataset_name + " image " + str(counter + 1) + "/" + str(image_count))

        # For each variation get the curve for the method
        # The three attribution methods are averaged at the end
        for i in range(len(grad_variations)):
            ######## score the attribuions ########
            _, MAS_ins_grad, _, _, RISE_ins_grad = MAS_insertion.single_run(ins_del_img, grad_variations[i], device, max_batch_size = batch_size)
            _, MAS_ins_lig, _, _, RISE_ins_lig = MAS_insertion.single_run(ins_del_img, LIG_variations[i], device, max_batch_size = batch_size)
            _, MAS_ins_gbp, _, _, RISE_ins_gbp = MAS_insertion.single_run(ins_del_img, GBP_variations[i], device, max_batch_size = batch_size)

            RISE_ins_curves[i] += RISE_ins_lig + RISE_ins_grad + RISE_ins_gbp
            MAS_ins_curves[i] += MAS_ins_lig + MAS_ins_grad + MAS_ins_gbp

            _, MAS_del_grad, _, _, RISE_del_grad = MAS_deletion.single_run(ins_del_img, grad_variations[i], device, max_batch_size = batch_size)
            _, MAS_del_ig, _, _, RISE_del_ig = MAS_deletion.single_run(ins_del_img, LIG_variations[i], device, max_batch_size = batch_size)
            _, MAS_del_gbp, _, _, RISE_del_gbp = MAS_deletion.single_run(ins_del_img, GBP_variations[i], device, max_batch_size = batch_size)

            RISE_del_curves[i] += RISE_del_ig + RISE_del_grad + RISE_del_gbp
            MAS_del_curves[i] += MAS_del_ig + MAS_del_grad + MAS_del_gbp
            
            sic_score_grad, aic_score_grad = PIC_Test.compute_both_metrics(AIC_SIC_img, grad_variations[i], random_mask, saliency_thresholds, model, device, transform_normalize)      
            sic_score_ig, aic_score_ig = PIC_Test.compute_both_metrics(AIC_SIC_img, LIG_variations[i], random_mask, saliency_thresholds, model, device, transform_normalize)      
            sic_score_gbp, aic_score_gbp = PIC_Test.compute_both_metrics(AIC_SIC_img, GBP_variations[i], random_mask, saliency_thresholds, model, device, transform_normalize)      

            SIC_curves[i] += sic_score_ig.curve_y + sic_score_grad.curve_y + sic_score_gbp.curve_y
            AIC_curves[i] += aic_score_ig.curve_y + aic_score_grad.curve_y + aic_score_gbp.curve_y

        print(time.time() - start)

        # when all tests have passed the number of images used can go up by 1
        counter += 1

    # divide all (3, n) curve storing arrays by the 3 atttributions per and num images per
    SIC_curves = SIC_curves / 3 / counter
    AIC_curves = AIC_curves / 3 / counter
    RISE_ins_curves = RISE_ins_curves / 3 / counter
    RISE_del_curves = RISE_del_curves / 3 / counter
    MAS_ins_curves = MAS_ins_curves / 3 / counter
    MAS_del_curves = MAS_del_curves / 3 / counter

    RISE_ins_curves = (RISE_ins_curves - RISE_ins_curves.min()) / (RISE_ins_curves.max() - RISE_ins_curves.min())
    RISE_del_curves = (RISE_del_curves - RISE_del_curves.min()) / (RISE_del_curves.max() - RISE_del_curves.min())

    # make the test folder if it doesn't exist
    folder = "../test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the noise ignorance test
    file_name = dataset_name + "_" + noise_type + "_sensitivity_" + str(image_count) + "_images"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerows(SIC_curves)
        write.writerows(AIC_curves)
        write.writerows(RISE_ins_curves)
        write.writerows(RISE_del_curves)
        write.writerows(MAS_ins_curves)
        write.writerows(MAS_del_curves)

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # img_hw determines how to transform input images for model needs
    if FLAGS.model == "R101":
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
        modified_model = resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")

        if FLAGS.dataset_name == "resisc45":
            check_name = '../../model_training/resisc_45_resnet101.pth.tar'
            model.fc = nn.Linear(model.fc.in_features, 45)
            modified_model.fc = nn.Linear(model.fc.in_features, 45)

    elif FLAGS.model == "VIT16":
        model = models.vit_b_16(weights = "ViT_B_16_Weights.IMAGENET1K_V1")
        modified_model = vit.vit_b_16(weights = "ViT_B_16_Weights.IMAGENET1K_V1")

        if FLAGS.dataset_name == "resisc45":
            check_name = '../../model_training/resisc_45_vit_b_16.pth.tar'
            model.heads.head = nn.Linear(model.heads.head.in_features, 45)
            modified_model.heads.head = nn.Linear(model.heads.head.in_features, 45)

    img_hw = 224
    batch_size = 50
    function_steps = 50

    checkpoint = torch.load(check_name, map_location = device)
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    if FLAGS.model == "R101": 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        modified_model.load_state_dict(new_state_dict)
    elif FLAGS.model == "VIT16":
        model.load_state_dict(state_dict)
        modified_model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    modified_model.to(device)
    modified_model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_hw, img_hw)),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])


    random_mask = PIC_Test.generate_random_mask(img_hw, img_hw, .01)
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

    run_and_save_tests(FLAGS.noise_type, img_hw, transform, random_mask, saliency_thresholds, FLAGS.image_count, function_steps, batch_size, model, modified_model, FLAGS.model, device, FLAGS.dataset_path, FLAGS.dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Test the ignorance to noise of the different metrics.')
    parser.add_argument('--image_count',
                        type = int, default = 1000,
                        help='How many images to test with.')
    parser.add_argument('--noise_type',
                        type = str, default = "constant",
                        help='constant or noised')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101 or VIT16')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--dataset_name',
                type = str, default = "resisc45",
                help='Supported datasets: resisc45')
    parser.add_argument('--dataset_path',
            type = str, default = "../../resisc45",
            help = 'The path to your dataset input')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import os
import csv
import time
import argparse
import numpy as np

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from captum.attr import GuidedBackprop
from util.attribution_methods import saliencyMethods as attribution
from util.test_methods import MASTestFunctions as MAS
from util.test_methods import sanityForMetrics as SFMetric
from util.test_methods import PICTestFunctions as PIC_Test
from util.modified_models import resnet

from util.attribution_methods.VIT_LRP.interpret_methods import InterpretTransformer
from util.attribution_methods.VIT_LRP.ViT_new import vit_base_patch16_224


model = None

# standard resisc45 normalization
transform_normalize = transforms.Normalize(
    mean=[0.368, 0.381, 0.344],
    std=[0.200, 0.181, 0.181]
)

transform_normalize_VIT = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

def run_and_save_tests(img_hw, transform, random_mask, saliency_thresholds, image_count, function_steps, batch_size, model, modified_model, it, model_name, device, dataset_path, dataset_name):
    # initialize RISE and MAS blur kernel
    klen = 11
    ksig = 11
    kern = MAS.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

    MAS_insertion = MAS.MASMetric(model, img_hw * img_hw, 'ins', img_hw, substrate_fn = blur)
    MAS_deletion = MAS.MASMetric(model, img_hw * img_hw, 'del', img_hw, substrate_fn = torch.zeros_like)

    if model_name == "R101":
        normalize = transform_normalize
    elif model_name == "VIT16":
        normalize = transform_normalize_VIT

    if dataset_name == "resisc45":
        num_classes = 45

    images_per_class = int(np.ceil(image_count / num_classes))
    classes_used = [0] * num_classes

    RISE_ins_scores = []
    RISE_del_scores = []
    RISE_ins_del_scores = []
    MAS_ins_scores = []
    MAS_del_scores = []
    MAS_ins_del_scores = []
    SIC_scores = []
    AIC_scores = []

    RISE_ins_order = []
    RISE_del_order = []
    RISE_ins_del_order = []
    MAS_ins_order = []
    MAS_del_order = []
    MAS_ins_del_order = []
    SIC_order = []
    AIC_order = []

    attribution_maps = []
    zero_attr_flag = 0

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

        trans_img = img.squeeze()
        # only rgb images can be classified
        if trans_img.shape != (3, img_hw, img_hw):
            continue

        tensor_img = normalize(trans_img)
        tensor_img = torch.unsqueeze(tensor_img, 0)

        ins_del_img = tensor_img
        PIC_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))

        # tensor_img is in form needed for prediction for the ins/del method
        ins_del_img = tensor_img

        # check if classification is correct
        model_pred = model_utils.getClass(tensor_img, model, device)
        if model_pred.cpu() != target_class:
            continue
    
        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       

        if model_name == "R101":
            ########  Gradient  ########
            gradient = model_utils.getGradients(tensor_img, model, device, target_class)
            saliency_map = np.transpose(gradient.squeeze().detach().cpu().numpy(), (1, 2, 0))
            attribution_maps.append(np.abs(np.sum(saliency_map, axis = 2)))

            ########  LIG  ########
            lig = attribution.IG(tensor_img, model, function_steps, batch_size, 0.9, 0, device, target_class) 
            saliency_map = np.transpose(lig.squeeze().detach().cpu().numpy(), (1, 2, 0))
            attribution_maps.append(np.abs(np.sum(saliency_map, axis = 2)))

            ########  Guided Backprop  ########
            guided_bp = GuidedBackprop(modified_model)
            tensor_img.requires_grad = True
            saliency_map = guided_bp.attribute(tensor_img.to(device), target = target_class)
            tensor_img.requires_grad = False
            saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))
            attribution_maps.append(np.abs(np.sum(saliency_map, axis = 2)))
        elif model_name == "VIT16":
            ########  Random  ########
            saliency_map = torch.rand_like(tensor_img)
            saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))
            attribution_maps.append(np.abs(np.sum(saliency_map, axis = 2)))

            ########  LIG  ########
            lig = attribution.IG(tensor_img, model, function_steps, batch_size, 0.9, 0, device, target_class) 
            saliency_map = np.transpose(lig.squeeze().detach().cpu().numpy(), (1, 2, 0))
            attribution_maps.append(np.abs(np.sum(saliency_map, axis = 2)))

            ########  raw attention  ########
            pred = it.raw_attn(tensor_img.to(device)).reshape(14, 14)
            mask = torch.nn.functional.interpolate(pred.reshape(1, 1, pred.shape[0], pred.shape[1]), scale_factor=16, mode='bilinear')
            mask = mask.reshape((1, 224, 224)) * torch.ones((3, 224, 224)).to(device)
            saliency_map = np.transpose(mask.squeeze().detach().cpu().numpy(), (1, 2, 0))
            attribution_maps.append(np.abs(np.sum(saliency_map, axis = 2)))


        # check if any attribution maps are invalid
        for attribution_map in attribution_maps:
            if np.sum(attribution_map.reshape(1, 1, img_hw ** 2)) == 0:
                print("Skipping Image due to 0 attribution in a method")
                zero_attr_flag = 1
                break

        if zero_attr_flag == 1:
            classes_used[target_class] -= 1
            attribution_maps.clear()
            zero_attr_flag = 0
            continue

        print(model_name + " metric consistency test, " + dataset_name + " image " + str(counter + 1) + "/" + str(image_count))

        num_attrs = len(attribution_maps)

        score_list_RISE_ins = [0] * num_attrs
        score_list_RISE_del = [0] * num_attrs
        score_list_MAS_ins = [0] * num_attrs
        score_list_MAS_del = [0] * num_attrs
        score_list_SIC = [0] * num_attrs
        score_list_AIC = [0] * num_attrs

        start = time.time()
        # capture scores for the attributions
        for i in range(num_attrs):
            _, MAS_ins, _, _, RISE_ins = MAS_insertion.single_run(ins_del_img, attribution_maps[i], device, max_batch_size = batch_size)
            _, MAS_del, _, _, RISE_del = MAS_deletion.single_run(ins_del_img, attribution_maps[i], device, max_batch_size = batch_size)
            sic_score, aic_score = PIC_Test.compute_both_metrics(PIC_img, attribution_maps[i], random_mask, saliency_thresholds, model, device, normalize)   

            score_list_RISE_ins[i] = MAS.auc(RISE_ins)
            score_list_RISE_del[i] = MAS.auc(RISE_del)
            score_list_MAS_ins[i] = MAS.auc(MAS_ins)
            score_list_MAS_del[i] = MAS.auc(MAS_del)
            score_list_SIC[i] = sic_score.auc
            score_list_AIC[i] = aic_score.auc

        score_list_RISE_ins_del = np.asarray(score_list_RISE_ins) - np.asarray(score_list_RISE_del)
        score_list_MAS_ins_del = np.asarray(score_list_MAS_ins) - np.asarray(score_list_MAS_del)

        RISE_ins_scores.append(score_list_RISE_ins)
        RISE_del_scores.append(score_list_RISE_del)
        RISE_ins_del_scores.append(score_list_RISE_ins_del)
        MAS_ins_scores.append(score_list_MAS_ins)
        MAS_del_scores.append(score_list_MAS_del)
        MAS_ins_del_scores.append(score_list_MAS_ins_del)
        SIC_scores.append(score_list_SIC)
        AIC_scores.append(score_list_AIC)

        # find the oderings of the saliency maps
        RISE_ins_order.append(np.argsort(RISE_ins_scores[counter])[::-1])
        RISE_del_order.append(np.argsort(RISE_del_scores[counter]))
        RISE_ins_del_order.append(np.argsort(RISE_ins_del_scores[counter])[::-1])
        MAS_ins_order.append(np.argsort(MAS_ins_scores[counter])[::-1])
        MAS_del_order.append(np.argsort(MAS_del_scores[counter]))
        MAS_ins_del_order.append(np.argsort(MAS_ins_del_scores[counter])[::-1])
        SIC_order.append(np.argsort(SIC_scores[counter])[::-1])
        AIC_order.append(np.argsort(AIC_scores[counter])[::-1])

        print(time.time() - start)

        # when all tests have passed the number of images used can go up by 1 and clear map list
        counter += 1
        attribution_maps.clear()

    RISE_ins_IRR = SFMetric.inter_rater_reliability(np.asarray(RISE_ins_order))
    RISE_del_IRR = SFMetric.inter_rater_reliability(np.asarray(RISE_del_order))
    RISE_ins_del_IRR = SFMetric.inter_rater_reliability(np.asarray(RISE_ins_del_order))
    MAS_ins_IRR = SFMetric.inter_rater_reliability(np.asarray(MAS_ins_order))
    MAS_del_IRR = SFMetric.inter_rater_reliability(np.asarray(MAS_del_order))
    MAS_ins_del_IRR = SFMetric.inter_rater_reliability(np.asarray(MAS_ins_del_order))
    SIC_IRR = SFMetric.inter_rater_reliability(np.asarray(SIC_order))
    AIC_IRR = SFMetric.inter_rater_reliability(np.asarray(AIC_order))

    RISE_ICR = SFMetric.internal_consistency_reliability(np.asarray(RISE_ins_order), np.asarray(RISE_del_order))
    MAS_ICR = SFMetric.internal_consistency_reliability(np.asarray(MAS_ins_order), np.asarray(MAS_del_order))
    PIC_ICR = SFMetric.internal_consistency_reliability(np.asarray(SIC_order), np.asarray(AIC_order))

    # make the test folder if it doesn't exist
    folder = "../test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the test results
    file_name = dataset_name + "_consistency_test_" + str(image_count) + "_images"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["SIC IRR", str(SIC_IRR)])
        write.writerow(["AIC IRR", str(AIC_IRR)])
        write.writerow("")

        write.writerow(["RISE ins IRR", str(RISE_ins_IRR)])
        write.writerow(["RISE del IRR", str(RISE_del_IRR)])
        write.writerow(["RISE diff IRR", str(RISE_ins_del_IRR)])
        write.writerow("")

        write.writerow(["MAS ins IRR", str(MAS_ins_IRR)])
        write.writerow(["MAS del IRR", str(MAS_del_IRR)])
        write.writerow(["MAS diff IRR", str(MAS_ins_del_IRR)])
        write.writerow("")

        write.writerow(["PIC ICR", str(PIC_ICR)])
        write.writerow(["RISE ICR", str(RISE_ICR)])
        write.writerow(["MAS ICR", str(MAS_ICR)])

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

        checkpoint = torch.load(check_name, map_location = device)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        modified_model.load_state_dict(new_state_dict)

        model.eval()
        modified_model.eval()
        model.to(device)
        modified_model.to(device)

        it = None

    elif FLAGS.model == "VIT16":
        model = vit_base_patch16_224(pretrained=True).to(device)
        modified_model = None

        if FLAGS.dataset_name == "resisc45":
            check_name = "../../model_training/resisc_45_vit_b_16_normal.pth.tar"
            model.head = nn.Linear(model.head.in_features, 45)

        checkpoint = torch.load(check_name, map_location = device)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        model.eval()
        model.to(device)

        it = InterpretTransformer(model, device)
    else:
        print("Please select a model from: R101 or VIT16")
        exit()

    img_hw = 224
    batch_size = 50
    function_steps = 50

    transform = transforms.Compose([
        transforms.Resize((img_hw, img_hw)),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    random_mask = PIC_Test.generate_random_mask(img_hw, img_hw, .01)
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

    run_and_save_tests(img_hw, transform, random_mask, saliency_thresholds, FLAGS.image_count, function_steps, batch_size, model, modified_model, it, FLAGS.model, device, FLAGS.dataset_path, FLAGS.dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Consistency check the different metrics.')
    parser.add_argument('--image_count',
                        type = int, default = 1000,
                        help='How many images to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101 or VIT32')
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
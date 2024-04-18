import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import csv
import time
import argparse
import numpy as np

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from captum.attr import GuidedBackprop
from util.attribution_methods import saliencyMethods as attribution
from util.attribution_methods import AGI as AGI
from util.test_methods import MASTestFunctions as MAS
from util.test_methods import PICTestFunctions as PIC_Test
from util.test_methods import sanityForMetrics as SFMetric
from util.modified_models import resnet

from util.attribution_methods.VIT_LRP.interpret_methods import InterpretTransformer
from util.attribution_methods.VIT_LRP.ViT_new import vit_base_patch16_224


model = None

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
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

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    images_used = 0

    images_per_class = int(np.ceil(image_count / 1000))
    classes_used = [0] * 1000

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

    # look at test images in order from 1
    for image in sorted(os.listdir(dataset_path)):    
        if images_used == image_count:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        image_path = dataset_path + "/" + image
        PIL_img = Image.open(image_path)

        # put the image in form needed for prediction
        trans_img = transform(PIL_img)
        tensor_img = normalize(trans_img)
        tensor_img = torch.unsqueeze(tensor_img, 0)

        # only rgb images can be classified
        if trans_img.shape != (3, img_hw, img_hw):
            continue

        target_class = model_utils.getClass(tensor_img, model, device)

        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       
        
        print(model_name + " metric consistency test, image: " + image + " " + str(images_used + 1) + "/" + str(image_count))

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

            ########  trans attn  ########
            pred = it.transition_attention_maps(tensor_img.to(device), start_layer = 0, steps = 20).reshape(14, 14)
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

        # Get attribution scores
        ins_del_img = tensor_img
        PIC_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))

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
        RISE_ins_order.append(np.argsort(RISE_ins_scores[images_used])[::-1])
        RISE_del_order.append(np.argsort(RISE_del_scores[images_used]))
        RISE_ins_del_order.append(np.argsort(RISE_ins_del_scores[images_used])[::-1])
        MAS_ins_order.append(np.argsort(MAS_ins_scores[images_used])[::-1])
        MAS_del_order.append(np.argsort(MAS_del_scores[images_used]))
        MAS_ins_del_order.append(np.argsort(MAS_ins_del_scores[images_used])[::-1])
        SIC_order.append(np.argsort(SIC_scores[images_used])[::-1])
        AIC_order.append(np.argsort(AIC_scores[images_used])[::-1])

        print(time.time() - start)

        # when all tests have passed the number of images used can go up by 1 and clear map list
        images_used += 1
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
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device)
        
        modified_model = resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device)
        modified_model = modified_model.eval()

        it = None

    elif FLAGS.model == "VIT16":
        model = vit_base_patch16_224(pretrained=True).to(device)

        modified_model = None
        
        it = InterpretTransformer(model, device)
    else:
        print("Please select a model from: R101 or VIT16")
        exit()
        
    model = model.eval()

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
    parser = argparse.ArgumentParser('Consistenty check the different metrics.')
    parser.add_argument('--image_count',
                        type = int, default = 1000,
                        help='How many images to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101 or VIT16')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--dataset_name',
                type = str, default = "ImageNet")
    parser.add_argument('--dataset_path',
            type = str, default = "../../ImageNet",
            help = 'The path to your dataset input')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
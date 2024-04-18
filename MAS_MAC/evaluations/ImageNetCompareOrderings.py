import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from skimage.transform import resize
from captum.attr import IntegratedGradients, LayerGradCam, GuidedBackprop, GuidedGradCam

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.attribution_methods import saliencyMethods as attribution
from util.attribution_methods import GIGBuilder as GIG_Builder
from util.attribution_methods import AGI as AGI
from util.test_methods import MASTestFunctions as MAS
from util.test_methods import PICTestFunctions as PIC_test
from util.modified_models import resnet
from util.visualization import attr_to_subplot

model = None

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# runs an attribution method w 3 baselines over imageCount images and calculates the mean PIC
def run_and_save_tests(img_hw, transform_list, random_mask, saliency_thresholds, image_count, function_steps, batch_size, model, modified_model, model_name, device, imagenet):
    # make the test folder if it doesn't exist
    folder = "../test_results/order_comps/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # set up new AGI model
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm_layer = AGI.Normalize(mean, std)
    AGI_model = nn.Sequential(norm_layer, model).to(device)

    # initialize RISE blur kernel
    klen = 11
    ksig = 11
    kern = MAS.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

    new_insertion = MAS.MASMetric(model, img_hw * img_hw, 'ins', img_hw, substrate_fn = blur)
    new_deletion = MAS.MASMetric(model, img_hw * img_hw, 'del', img_hw, substrate_fn = torch.zeros_like)

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    labels_path = '../../util/class_maps/ImageNet/imagenet_classes.txt'
    with open(labels_path) as f:
        classes = [line.strip() for line in f.readlines()]

    images_used = 0

    images_per_class = int(np.ceil(image_count / 1000))
    classes_used = [0] * 1000

    attributions = [0] * 7
    attribution_names = ["IG", "LIG", "GIG", "AGI", "GC", "GGC", "GBP"]

    RISE_ins_scores = np.zeros((image_count, len(attributions)))
    RISE_del_scores = np.zeros((image_count, len(attributions)))
    MAS_ins_scores = np.zeros((image_count, len(attributions)))
    MAS_del_scores = np.zeros((image_count, len(attributions)))
    SIC_scores = np.zeros((image_count, len(attributions)))
    AIC_scores = np.zeros((image_count, len(attributions)))      

    # look at test images in order from 1
    for image in sorted(os.listdir(imagenet)):    
        if images_used == image_count:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        image_path = imagenet + "/" + image

        PIL_img = Image.open(image_path)
        img = transform_list[0](PIL_img)

        # put the image in form needed for IG, LIG, GC, GGC, GBP
        attr_img = transform_normalize(img)
        attr_img = torch.unsqueeze(attr_img, 0)

        # only rgb images can be classified
        if img.shape != (3, img_hw, img_hw):
            continue

        target_class = model_utils.getClass(attr_img, model, device)

        class_name = classes[target_class.item()]

        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       

        print(model_name + ", image: " + image + " " + str(images_used + 1) + "/" + str(image_count))


        ########  IG  ########
        integrated_gradients = IntegratedGradients(model)
        saliency_map = integrated_gradients.attribute(attr_img.to(device), 0, target = target_class, n_steps = function_steps, internal_batch_size=batch_size)
        saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2))
        if np.sum(saliency_map_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue
        attributions[0] = saliency_map_test

        ########  LIG  ########
        saliency_map = attribution.IG(attr_img, model, function_steps, batch_size, .9, 0, device, target_class)
        saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2))
        if np.sum(saliency_map_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue
        attributions[1] = saliency_map_test

        ########  GIG  ########
        call_model_args = {'class_idx_str': target_class.item()}
        guided_ig = GIG_Builder.GuidedIG()
        baseline = torch.zeros_like(attr_img)
        saliency_map = guided_ig.GetMask(attr_img, model, device, GIG_Builder.call_model_function, call_model_args, x_baseline = baseline, x_steps = function_steps, max_dist = 1.0, fraction = 0.5)
        saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2))
        if np.sum(saliency_map_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue
        attributions[2] = saliency_map_test

        ########  AGI  ########
        epsilon = 0.05
        max_iter = 20
        topk = 1
        # define the ids of the selected adversarial class
        selected_ids = range(0, 999, int(1000 / topk)) 
        agi_img = AGI.LoadImage(image_path, transform_list[1], transform_list[2])
        agi_img = agi_img.astype(np.float32) 
        # Run test
        example = AGI.test(AGI_model, device, agi_img, epsilon, topk, selected_ids, max_iter)
        AGI_map = example[2]
        if type(AGI_map) is not np.ndarray:
            print("AGI failure, skipping image")
            classes_used[target_class] -= 1
            continue
        saliency_map = np.transpose(AGI_map, (1, 2, 0))
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2))
        if np.sum(saliency_map_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue
        attributions[3] = saliency_map_test

        ########  Grad Cam  ########
        if model_name != "VIT32":
            layer = model.layer4
        else:
            layer = model.conv_proj
        layer_gc = LayerGradCam(model, layer)
        attr_img.requires_grad = True
        gc = layer_gc.attribute(attr_img.to(device), target_class, relu_attributions=True)
        gc = resize(gc.squeeze().cpu().detach().numpy(), (224, 224), preserve_range=True)
        gc = torch.tensor(gc).reshape((1, 224, 224)) * torch.ones((3, 224, 224))
        attr_img.requires_grad = False
        saliency_map = np.transpose(gc.squeeze().detach().cpu().numpy(), (1, 2, 0))
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2))
        if np.sum(saliency_map_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue
        attributions[4] = saliency_map_test

        ########  Guided Grad Cam  ########
        if model_name != "VIT32":
            guided_gc = GuidedGradCam(modified_model, modified_model.layer4)
        else:
            guided_gc = GuidedGradCam(modified_model, modified_model.conv_proj)
        attr_img.requires_grad = True
        ggc = guided_gc.attribute(attr_img.to(device), target_class)
        attr_img.requires_grad = False
        saliency_map = np.transpose(ggc.squeeze().detach().cpu().numpy(), (1, 2, 0))
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2))
        if np.sum(saliency_map_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue
        attributions[5] = saliency_map_test

        ######## GBP ########
        guided_bp = GuidedBackprop(modified_model)
        attr_img.requires_grad = True
        gbp = guided_bp.attribute(attr_img.to(device), target = target_class)
        attr_img.requires_grad = False
        saliency_map = np.transpose(gbp.squeeze().detach().cpu().numpy(), (1, 2, 0))
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2))
        if np.sum(saliency_map_test.reshape(1, 1, img_hw **2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue
        attributions[6] = saliency_map_test


        # put the image in form needed for prediction for the ins/del methods
        ins_del_img = transform_normalize(img)
        ins_del_img = torch.unsqueeze(ins_del_img, 0)

        # image for AIC/SIC test
        AIC_SIC_img = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))

        start = time.time()

        # capture scores for the attributions
        for i in range(len(attributions)):
        # for i in range(0):
            _, MAS_ins, _, _, RISE_ins = new_insertion.single_run(ins_del_img, attributions[i], device, max_batch_size = batch_size)
            _, MAS_del, _, _, RISE_del = new_deletion.single_run(ins_del_img, attributions[i], device, max_batch_size = batch_size)

            sic_score, aic_score = PIC_test.compute_both_metrics(AIC_SIC_img, attributions[i], random_mask, saliency_thresholds, model, device, transform_normalize)

            RISE_ins_scores[images_used][i] = MAS.auc(RISE_ins)
            RISE_del_scores[images_used][i] = MAS.auc(RISE_del)
            MAS_ins_scores[images_used][i] = MAS.auc(MAS_ins)
            MAS_del_scores[images_used][i] = MAS.auc(MAS_del)
            SIC_scores[images_used][i] = sic_score.auc
            AIC_scores[images_used][i] = aic_score.auc

        print(time.time() - start)

        RISE_ins_del_scores = np.asarray(RISE_ins_scores) - np.asarray(RISE_del_scores)
        MAS_ins_del_scores = np.asarray(MAS_ins_scores) - np.asarray(MAS_del_scores)


        SIC_order = np.argsort(SIC_scores[images_used])[::-1]
        AIC_order = np.argsort(AIC_scores[images_used])[::-1]
        RISE_ins_order =  np.argsort(RISE_ins_scores[images_used])[::-1]
        RISE_del_order = np.argsort(RISE_del_scores[images_used])
        RISE_ins_del_order = np.argsort(RISE_ins_del_scores[images_used])[::-1]
        MAS_ins_order = np.argsort(MAS_ins_scores[images_used])[::-1]
        MAS_del_order = np.argsort(MAS_del_scores[images_used])
        MAS_ins_del_order = np.argsort(MAS_ins_del_scores[images_used])[::-1]
        
        # save the attributions
        plt.rcParams.update({'font.size': 42})
        fig, axs = plt.subplots(8, 8, figsize = (28, 34))
        norm = "absolute"

        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[0, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[SIC_order][i].reshape((img_hw, img_hw, 1)), "Rank " + str(i + 1) + "\n" + np.array(attribution_names)[SIC_order][i], axs[0, i + 1], norm = norm)
            axs[0, 0].set_ylabel("SIC")
        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[1, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[AIC_order][i].reshape((img_hw, img_hw, 1)), np.array(attribution_names)[AIC_order][i], axs[1, i + 1], norm = norm)
            axs[1, 0].set_ylabel("AIC")
        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[2, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[RISE_ins_order][i].reshape((img_hw, img_hw, 1)), np.array(attribution_names)[RISE_ins_order][i], axs[2, i + 1], norm = norm)
            axs[2, 0].set_ylabel("Ins")
        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[3, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[RISE_del_order][i].reshape((img_hw, img_hw, 1)), np.array(attribution_names)[RISE_del_order][i], axs[3, i + 1], norm = norm)
            axs[3, 0].set_ylabel("Del")
        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[4, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[RISE_ins_del_order][i].reshape((img_hw, img_hw, 1)), np.array(attribution_names)[RISE_ins_del_order][i], axs[4, i + 1], norm = norm)
            axs[4, 0].set_ylabel("Ins - Del")
        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[5, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[MAS_ins_order][i].reshape((img_hw, img_hw, 1)), np.array(attribution_names)[MAS_ins_order][i], axs[5, i + 1], norm = norm)
            axs[5, 0].set_ylabel("Ins")
        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[6, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[MAS_del_order][i].reshape((img_hw, img_hw, 1)), np.array(attribution_names)[MAS_del_order][i], axs[6, i + 1], norm = norm)
            axs[6, 0].set_ylabel("Del")
        for i in range(len(attributions)):
            attr_to_subplot(img, class_name, axs[7, 0], original_image = True)
            attr_to_subplot(np.array(attributions)[MAS_ins_del_order][i].reshape((img_hw, img_hw, 1)), np.array(attribution_names)[MAS_ins_del_order][i], axs[7, i + 1], norm = norm)
            axs[7, 0].set_ylabel("Ins - Del")

        plt.subplots_adjust(wspace=0.1)
        plt.figure(fig)
        plt.savefig("../test_results/order_comps/image_" + f"{(images_used + 1):03d}" + ".png", bbox_inches='tight', transparent = "True", pad_inches = .05)
        fig.clear()
        plt.close(fig)

        # when all tests have passed the number of images used can go up by 1
        images_used += 1

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    model_name = "R101"
    model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
    modified_model = resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")

    # img_hw determines how to transform input images for model needs
    img_hw = 224
    batch_size = 50

    function_steps = 50

    model = model.eval()
    model.to(device)

    modified_model = modified_model.eval()
    modified_model.to(device)

    # specify the transforms needed
    resize = transforms.Resize((img_hw, img_hw))
    crop = transforms.CenterCrop(img_hw)

    transform_IG = transforms.Compose([
        transforms.Resize((img_hw, img_hw)),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    transform_list = (transform_IG, resize, crop)

    random_mask = PIC_test.generate_random_mask(img_hw, img_hw, .01)
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

    run_and_save_tests(img_hw, transform_list, random_mask, saliency_thresholds, FLAGS.image_count, function_steps, batch_size, model, modified_model, model_name, device, FLAGS.imagenet)

def threshold(saliency_map, lowerbound = 10, upperbound = 99, img_hw = 224):
    hm = saliency_map
    hm = np.mean(hm, axis=0)
    l = np.percentile(hm, lowerbound)
    u = np.percentile(hm, upperbound)

    hm[hm < l] = l
    hm[hm > u] = u

    hm = (hm - l)/(u - l)

    saliency_map = np.reshape(hm, (img_hw, img_hw, 1))
    
    return saliency_map

if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Create PNGs comparing quality metric orderings.')
    parser.add_argument('--image_count',
                        type = int, default = 50,
                        help='How many images to test with.')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--imagenet',
                type = str, default = "imagenet",
                help = 'The path to your 2012 imagenet validation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
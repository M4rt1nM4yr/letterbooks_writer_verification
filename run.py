import os
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sklearn

from src.model.writer_identifier import WriterIdentifier
from src.metric.writer_identification_metrics import evaluate
from src.data.nbb_github_wi_fetcher import load_data
from src.utils.report_false_predictions import report_false_predictions


def run(
        root_imgs,
        root_diplomatic,
        books_train,
        books_test,
        model_ckpt=None,
        preload_images=True,
        just_load_data = False,
        produce_report_of_false_predictions = False,
):
    writer_model = WriterIdentifier(
        descriptor_binarizer="otsu",
        descriptor_hellinger_norm=True,
        descriptor_sampler="sift",
        vlad_gmp=False,
        reduction_method="rpca",
        reduction_dim=512,
        vlad_multitude=5,
        vlad_n_clusters=100,
    )
    # load data
    imgs, labels, names, meta_dict = load_data(
        root_imgs,
        root_diplomatic,
        books_train,
        color_mode="L",
        preload_images=preload_images,
    )
    if books_train != books_test:
        imgs_test, labels_test, names_test, meta_dict_test = load_data(
            root_imgs,
            root_diplomatic,
            books_test,
            color_mode="L",
            preload_images=preload_images,
        )
    else:
        imgs_test, labels_test, names_test, meta_dict_test = imgs.copy(), labels, names, meta_dict

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Writer distribution for train books {books_train}: \n{[(u, c) for u, c in zip(unique_labels, counts)]}")
    if just_load_data:
        return None

    for l,name in zip(labels, names):
        assert isinstance(l, int), f"For {name} the label is: {l}"
    print("loaded data")
    if preload_images:
        size_check = imgs[0][0].size
        writer_model.fit(imgs)
    else:
        n_samples = 250
        picked_ids = np.random.choice(range(len(imgs)), min(n_samples,len(imgs)), replace=False)
        print("load images now")
        new_imgs = list()
        for i in tqdm(picked_ids):
            sample_imgs = list()
            item = meta_dict[names[i].split("_")[-1]]
            for img_name, img_region in zip(imgs[i],item["letter_regions"]):
                img = Image.open(os.path.join(root_imgs, names[i].split("_")[0], img_name))
                sample_imgs.append(img.crop(img_region[1]).convert("L"))
            new_imgs.append(sample_imgs)
        writer_model.fit(new_imgs)

    ### TESTING
    if preload_images:
        assert size_check == imgs_test[0][0].size
        desc_test = writer_model.transform_fast(imgs_test)
    else:
        desc_test = list()
        # imgs_test = list()
        for i in tqdm(range(len(names_test))):
        # for i in tqdm(range(10)):
            sample_imgs = list()
            item = meta_dict_test[names_test[i].split("_")[-1]]
            for img_name, img_region in zip(imgs_test[i],item["letter_regions"]):
                img = Image.open(os.path.join(root_imgs, names_test[i].split("_")[0], img_name))
                sample_imgs.append(img.crop(img_region[1]).convert("L"))
            # imgs_test.append(None)
            desc_test.append(writer_model.transform_item(sample_imgs)["encodings"])
        desc_test = np.vstack(desc_test)
    print("### EVALUATION FAST ###")
    evaluation_info = evaluate(desc_test, labels_test, print_results=True, topk=True)

    if isinstance(imgs[0][0], str):
        helper0 = list()
        for name, img in zip(names_test, imgs_test):
            helper1 = [os.path.join(root_imgs, name.split("_")[0], im) for im in img]
            helper0.append(helper1)
        imgs_test = helper0

    if produce_report_of_false_predictions:
        report_false_predictions(
            names_test,
            labels_test,
            imgs_test,
            evaluation_info["classification_results"],
            meta_dict=meta_dict_test,
        )


    # Initialize UMAP
    reducer = umap.UMAP()
    # Fit the model and transform the data
    embeddings = reducer.fit_transform(desc_test)

    reducer = umap.UMAP()
    embeddings_norm = reducer.fit_transform(sklearn.preprocessing.normalize(desc_test, norm="l2", axis=1))

    return embeddings, embeddings_norm, labels_test, names_test, imgs_test, meta_dict_test



if __name__ == "__main__":
    book = ["Band2","Band3","Band4","Band5"]
    root_imgs = "add root path here"
    root_diplomatic = "add root path to diplmatic labels here"
    assert os.path.isdir(root_imgs)
    
    run(
        root_imgs=root_imgs,
        root_diplomatic=root_diplomatic,
        books_train=[book],
        books_test=[book],
        just_load_data=False,
        produce_report_of_false_predictions=True,
    )
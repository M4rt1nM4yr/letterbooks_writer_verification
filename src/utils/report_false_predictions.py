import os
from io import BytesIO
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import PIL.Image


def report_false_predictions(
        names,
        labels,
        imgs,
        classification_results,
        filepath = "/tmp/nbb_wi_report/",
        max_img_width = 400,
        max_img_height = 600,
        meta_dict= None,
):
    # set classification result to true where there is just one label in labels from one categorie
    label_spread = np.unique(labels, return_counts=True)
    underrepresented_writers = list()
    for label, count in zip(label_spread[0], label_spread[1]):
        if count == 1:
            underrepresented_writers.append(label)
    for idx, label in enumerate(labels):
        if label in underrepresented_writers:
            classification_results[idx] = True

    classification_results = np.logical_not(classification_results)

    os.makedirs(filepath, exist_ok=True)
    print(f"wrongly classified images: \n {np.array(names)[classification_results]}")
    filename = os.path.join(filepath, "wrongly_classified_images.pdf")
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = list()

    # create document of wrongly classified samples
    names_wrong, labels_wrong, imgs_wrong = list(), list(), list()
    for idx, b in enumerate(classification_results):
        if b:
            names_wrong.append(names[idx])
            labels_wrong.append(labels[idx])
            imgs_wrong.append(imgs[idx])

    for name, label, img in zip(names_wrong, labels_wrong, imgs_wrong):
        elements.append(Paragraph(f"Letter ID: {name}, Writer ID: {label}", styles["Heading2"]))
        for idx, i in enumerate(img):
            # fix image size
            if isinstance(i, str):
                i = PIL.Image.open(i)
                img_region = meta_dict[name.split("_")[-1]]["letter_regions"][idx][1]
                i = i.crop(img_region)
            img_width, img_height = i.size
            if img_width > max_img_width:
                i = i.resize((max_img_width, int(max_img_width * img_height / img_width)))
            if i.size[1] > max_img_height:
                i = i.crop((0, 0, i.size[0], max_img_height))
            new_width, new_height = i.size
            img_io = BytesIO()
            i.save(img_io, format="PNG")
            img_io.seek(0)
            elements.append(Image(img_io, width=new_width, height=new_height))
            elements.append(Spacer(1, 12))

        elements.append(Spacer(1, 12))
        elements.append(PageBreak())

    doc.build(elements)

    print(f"PDF '{filename}' created successfully.")


if __name__ == "__main__":
    names = ["Band2_a", "Band2_b", "Band2_c", "Band2_d"]
    labels = [1, 2, 2, 1]
    imgs = list()
    for i in range(len(names)):
        helper = list()
        for j in range(3):
            new_shape = tuple(np.random.randint(size=(2,), low=100, high=1000))
            helper.append(PIL.Image.new("RGB", new_shape, color="green"))
        imgs.append(helper)
    imgs = [
        ["/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg", "/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg","/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg",],
        ["/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg",
         "/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg",
         "/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg", ],
        ["/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg",
         "/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg",
         "/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg", ],
        ["/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg",
         "/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg",
         "/cluster/mayr/nbb/vanilla/Band2/Rst Nbg-Briefbücher-Nr 2_0009_right.jpg", ],
    ]
    meta_dict = {
        "a": {"letter_regions": [(0, (0, 0, 500, 500)), (0, (0, 0, 100, 100)), (0, (0, 0, 100, 100))]},
        "b": {"letter_regions": [(0, (0, 0, 100, 100)), (0, (0, 0, 100, 100)), (0, (0, 0, 100, 100))]},
        "c": {"letter_regions": [(0, (0, 0, 100, 100)), (0, (0, 0, 100, 100)), (0, (0, 0, 100, 100))]},
        "d": {"letter_regions": [(0, (0, 0, 100, 100)), (0, (0, 0, 100, 100)), (0, (0, 0, 100, 100))]},
    }
    classification_results = np.array([False, True, False, True])
    report_false_predictions(np.array(names), np.array(labels), imgs, classification_results, meta_dict=meta_dict)
    # expected output: "wrongly classified images: \n ['b', 'd']"
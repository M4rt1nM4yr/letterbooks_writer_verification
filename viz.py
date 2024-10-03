import os
import pickle, gzip
from bokeh import palettes
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import numpy as np
from bokeh.models import HoverTool

from run import run

def produce_bokeh_plot(
        embeddings,
        labels,
        names,
        books,
):

    source = ColumnDataSource(
        data=dict(
            Y0=embeddings[:, 0],
            Y1=embeddings[:, 1],
            colors_rgb=_get_colors(labels),
            names=names,
            labels=labels,
        )
    )

    hover = HoverTool(
        tooltips=[
            ("Sample Name", "@names"),
            ("LabelID", "@labels")
        ],
    )
    p = figure(tools=[hover, "crosshair", "zoom_in", "zoom_out", "box_zoom", "redo", "reset"], title="Pages overview",
               plot_width=900, plot_height=900)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.scatter('Y0', 'Y1', fill_color="colors_rgb", size=14, fill_alpha=0.8, source=source)

    output_file(f"nbb_writer_identification_{'_'.join(books)}.html",
                title=f"NBB Writer Identification Books: {' '.join(books)}")

    return p

def _get_colors(labels):
    colors = np.array(palettes.Category20[20])
    colors_rgb = np.array([tuple(int(colors[i][j:j + 2], 16) for j in (1, 3, 5)) for i in range(len(colors))])
    colors_rgb = colors_rgb / 255.

    colors_rgb = colors_rgb[labels]
    return colors_rgb

def get_data(
        root_imgs,
        root_diplomatic,
        books_train,
        books_test,
        preload_images=True,
        overwrite=False,
):
    filename = "_".join(books_test)+".pkl.gz"
    path = os.path.join("data",filename)
    if not os.path.isfile(path) or overwrite:
        print("fetching data")
        embeddings, embeddings_norm, labels, names, imgs, meta_dict = run(
            root_imgs,
            root_diplomatic,
            books_train,
            books_test,
            preload_images=preload_images,
            produce_report_of_false_predictions=True,
        )
        with gzip.open(path, "wb") as f:
            pickle.dump((embeddings, embeddings_norm, labels, names, imgs, meta_dict), f)
    else:
        with gzip.open(path, "rb") as f:
            embeddings, embeddings_norm, labels, names, imgs, meta_dict = pickle.load(f)

    return embeddings, embeddings_norm, labels, names, imgs, meta_dict


if __name__ == "__main__":
    books_all = [
        ["Band2", "Band3", "Band4", "Band5"],
        ["Band2", "Band3", "Band4", "Band5"],
    ]
    books_2 = [
        ["Band3", "Band4", "Band5"],
        ["Band2"],
    ]
    books_3 = [
        ["Band2", "Band4", "Band5"],
        ["Band3"],
    ]
    books_4 = [
        ["Band2", "Band3", "Band5"],
        ["Band4"],
    ]
    books_5 = [
        ["Band2", "Band3", "Band4"],
        ["Band5"],
    ]
    books = books_all

    root_imgs = "add root path here"
    root_diplomatic = "add root path to diplmatic labels here"
    
    embeddings, embeddings_norm, labels, names, imgs, meta_dict = get_data(
        root_imgs=root_imgs,
        root_diplomatic=root_diplomatic,
        books_train=books[0],
        books_test=books[1],
        overwrite=False,
        preload_images=False,
    )
    p = produce_bokeh_plot(
        embeddings,
        labels,
        names,
        books=books[1],
    )
    show(p)
    p_norm = produce_bokeh_plot(
        embeddings_norm,
        labels,
        names,
        books=books[1],
    )
    show(p_norm)
    print("finished")

# %%
import matplotlib.pyplot as plt

# %%
def plot_image_pair(ir_image_tensor, vis_image_tensor, output_file=None, normalized=False):
    """plots an IR and VIS image pair next to each other

    Parameters
    ----------
    ir_image_tensor (tf.tensor): IR image
    vis_image_tensor (tf.tensor): VIS image
    output_file (str, optional): save plot to this output file, by default None
    normalized (bool, optional): states if images are normalized or not, by default False
    """
    vmin = -1 if normalized else 0
    vmax = 1 if normalized else 255

    fig, axes = plt.subplots(1,2, figsize = (7,3))
    # plot IR image
    axes[0].imshow(ir_image_tensor, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("IR image")
    axes[0].set_axis_off()
    # plot VIS image
    axes[1].imshow(vis_image_tensor, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("VIS image")
    axes[1].set_axis_off()
    # save image if output file is defined
    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()

# %% saving intermediate states during training
def plot_images_at_epoch(ir_image, fake_vis_image, real_vis_image, output_file=None, normalized=True):
    """plots an IR, fake VIS and real VIS image triplet next to each other

    Parameters
    ----------
    ir_image (tf.tensor): IR image
    fake_vis_image (tf.tensor): fake VIS image
    real_vis_image (tf.tensor): real VIS image
    output_file (str, optional): save plot to this output file, by default None
    normalized (bool, optional): states if images are normalized or not, by default True
    """
    
    # if images have been normalized the range is [-1..1] else [0..255]
    vmin = -1 if normalized else 0
    vmax = 1 if normalized else 255

    plt.figure(figsize = (15,15))
    display_list= [ir_image, fake_vis_image, real_vis_image]
    title = ["IR image", "fake VIS image", "real VIS image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i], cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
    # save image if output file is defined
    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


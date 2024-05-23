# %%
import matplotlib.pyplot as plt

# %%
def plot_image_pair(ir_image_tensor, vis_image_tensor, output_file=None):

    fig, axes = plt.subplots(1,2, figsize = (7,3))
    # plot IR image
    axes[0].imshow(ir_image_tensor, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("IR image")
    axes[0].set_axis_off()
    # plot VIS image
    axes[1].imshow(vis_image_tensor, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("VIS image")
    axes[1].set_axis_off()
    # save image if output file is defined
    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()

# %% saving intermediate states during training
def plot_images_at_epoch(ir_image, predict_image, vis_image, output_file=None):

    plt.figure(figsize = (15,15))
    display_list= [ir_image, predict_image, vis_image]
    title = ["IR image", "predicted VIS image", "true VIS image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    # save image if output file is defined
    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()

# %%

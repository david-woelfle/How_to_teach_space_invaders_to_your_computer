"""
Some functions that came in handy while working on the project.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import matplotlib.animation as animation
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Activation

def widen_laser_shots(observation, widen_filter_shape=(7, 7)):
    mean_channel_value = observation.mean(axis=-1)
    coordinates_of_lasers_mask_2d = mean_channel_value > 139
    widen_filter = np.ones(widen_filter_shape)
    extended_lasers_mask = convolve2d(coordinates_of_lasers_mask_2d, 
                                      widen_filter, 'same').astype(bool)

    extended_lasers_mask_3d = np.stack([extended_lasers_mask]*3, axis=-1)

    observation[extended_lasers_mask_3d] = 255

    return observation

def create_test_model_for_agent():
    """
    Creates a model for testing that has similar in and output
    as the trained encoder.
    """
    input_layer = Input((256, 192, 3), name='input')
    x = Activation('linear')(input_layer)
    x = Flatten()(x)
    x = Dense(64, name='hidden')(x)
    output = Dense(6, activation='softmax')(x)
    model = Model(input_layer, output)
    model.summary()
    model.compile(loss='mae', optimizer='rmsprop')
    return model

def animate_episode(observations):
    """
    Convert the observations of the episode into an animation.
    
    Arguments:
    ----------
    observations: List or array of RGB Arrays
        The RGB values of the pixels to animate.
    
    Returns:
    --------
    ani: matplotlib animation object
        Save with ani.save('/your/path/file.mp4')
    """
    fig = plt.figure(figsize=(8, 10.5))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    frame_images = []
    for frame in observations:
        frame_image = plt.imshow(frame, animated=True)
        frame_images.append([frame_image])
    plt.close()

    ani = animation.ArtistAnimation(fig, 
                                    frame_images, 
                                    interval=40, 
                                    blit=True)

    return ani

def save_vae_test(vae, test_frame, fig_fn):
    """
    Create a test image to inspect the auto encoder training state.
    
    The image is showing the test frame after encoding and decoding
    it with the auto encoder.
    
    Arguments:
    ----------
    enoder: object
        The encoder part of the vae object below.
    decoder: object
        The decoder part of the vae object below.
    test_frame: 3d array
        image data, that is array with shape (y, x, channel)
    fig_fn: str
        Filename where to save the plot to.
    variational_mode: bool
        If the autoencoder is in variational mode.
    """
    # Create the plot and remove ticks as they have no use.
    fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticks([])

    # En- and Decode the test frame
    test_frame_as_float = np.expand_dims(test_frame, 0).astype('float32')/255 
    test_frame_decoded = vae.predict(test_frame_as_float)[0]

    ax1.set_title('original picture')
    ax1.imshow(test_frame)

    ax2.set_title('after VAE')
    ax2.imshow(test_frame_decoded)

    fig.savefig(fig_fn)
    plt.close()

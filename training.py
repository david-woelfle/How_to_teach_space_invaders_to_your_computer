import os
import sys
import logging
from datetime import datetime as dt

import cma
import numpy as np
import pandas as pd
from keras.losses import mse
from keras import backend as K
from keras.optimizers import RMSprop
from keras.models import Model, load_model
from keras.layers import Dense, Input, Lambda, Conv2D 
from keras.layers import Flatten, Reshape, Conv2DTranspose

from helpers import save_vae_test, widen_laser_shots
from extractor import ExtractWorker, KerasModelPolicy, EpochObsProvider

def build_agent_model(input_shape):
    """
    Creates a Keras Model for the agent.
    
    Arguments:
    ----------
    input_shape: tuple of int
        The shape of the output to which the model will be 
        connected. Without leading None.
    
    Returns:
    --------
    agent_model: Keras Model
        You guess what.
    """
    agent_input = Input(input_shape)
    agent_output = Dense(6, activation='tanh')(agent_input)
    agent_model = Model(agent_input, agent_output, name='agent')
    agent_model.summary(print_fn=logging.info)
    return agent_model

def flat_weights_to_model_format(weights_slices, weights_shapes, weights_flat):
    """
    Convert flat weights into shape matching model.set_weights
    """
    weights = []
    for weights_slice, weights_shape in zip(weights_slices, weights_shapes):
        layer_weights_flat = weights_flat[weights_slice]
        layer_weights = layer_weights_flat.reshape(weights_shape)
        weights.append(layer_weights)
    return weights

def run_agent_training(run_name,
                       probabilistic_mode=True,
                       cma_population_size=50,
                       cma_initial_std=0.05,
                       cma_max_generations=None,
                       **not_used_kwargs):
    """
    Executes the training of the agent.
    
    This will create lot's of data into results/{}.format(run_name).
    Will also log training process to stdout and agent_training.log file.
    
    Arguments:
    ----------
    run_name: str
        Identifier of the current experiement, is also the folder for output.
    probabilistic_mode: bool
        See extractor.KerasModelPolicy
    cma_population_size: int
        Population size per generation for CMA Optimizer
    cma_initial_std: float
        Initial standard deviation of CMA. Will be adapted by
        CMA while the optimization goes on.
    cma_max_generations: int or None
        Maximum number of generations to compute.
        No limit if None
    """
    # Create folders for intermediate results.
    RESULTS_DIR = 'results/{}'.format(run_name)
    AGENT_MODEL_CHECKPOINTS_DIR = '{}/agent_model_checkpoints'.format(RESULTS_DIR)
    AGENT_CMA_CHECKPOINTS_DIR = '{}/agent_cma_checkpoints'.format(RESULTS_DIR)
    for dirname in [AGENT_MODEL_CHECKPOINTS_DIR, AGENT_CMA_CHECKPOINTS_DIR]:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    # Configure logger to log to notebook and file
    log = logging.getLogger('tensorflow')
    log.handlers = []
    file_handler = logging.FileHandler("{}/agent_training.log".format(RESULTS_DIR))
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(format='%(asctime)s %(levelname)s:     %(message)s ',
                    handlers=[file_handler, stream_handler],
                    level=logging.INFO)

    log.info('\n\n\n')
    log.info('Started New Agent Training run.')

    ew = ExtractWorker(env_name='SpaceInvaders-v4')

    # Ignore the warning, we don't need the model compiled as
    # we won't use keras to train it anymore nor the agent.
    encoder_model = load_model('{}/encoder_model.h5'.format(RESULTS_DIR))

    # If trained in variational mode the output will be a list of
    # 3 layers. We only require z, the last of the three.
    if isinstance(encoder_model.output_shape, list):
        new_out = Lambda(lambda x: x[2])(encoder_model.output)
        encoder_model = Model(encoder_model.inputs, new_out)

    # Create the actual agent model.
    agent_input_shape = encoder_model.output_shape[1:]
    agent_model = build_agent_model(agent_input_shape)

    # Preserve information to restore the agent_model's weights
    # shape from a flat array.
    weights_slices = []
    weights_shapes = []
    agent_weights = agent_model.get_weights()
    slice_start = 0 
    for weight_array in agent_weights:
        weights_shapes.append(weight_array.shape)
        slice_end = slice_start + len(weight_array.ravel())
        weights_slices.append(slice(slice_start, slice_end))
    no_of_weights = sum(sl.stop for sl in weights_slices)

    # Assume that NN weights are centred around 0 with std=1 as inital
    # solution. Consider that for cma the best solution should be within
    # 3 * sigma0 around x0
    es = cma.CMAEvolutionStrategy(np.zeros(no_of_weights), 
                                  cma_initial_std,
                                  {'popsize': cma_population_size})
    generation = 0
    training_statistics_pre_df = {}
    while not es.stop():
        generation += 1
        start_time = dt.utcnow()

        weights_canidates = es.ask()
        log.info('')
        log.info('Evaluating {} canidates of generation {}'
                 .format(len(weights_canidates), generation))

        rewards = []
        for weights_canidate_flat in weights_canidates:
            weights_canidate = flat_weights_to_model_format(weights_slices, 
                                                            weights_shapes,
                                                            weights_canidate_flat)
            agent_model.set_weights(weights_canidate)

            # Connect the encoder with the agent to one model.
            inputs = encoder_model.inputs
            outputs = agent_model(encoder_model.outputs)
            full_model = Model(inputs, outputs)

            # Evaluate the performance of the canidate weights.
            policy_kw_args = {'model': full_model,
                              'probabilistic_mode': probabilistic_mode}
            # This function calls requires ~99% of the CPU time of this loop.
            # CPU time rises roughly linearly with n_episodes.
            stats = ew.extract_episode_statistics(policy_class=KerasModelPolicy, 
                                                  n_episodes=3, 
                                                  policy_kw_args=policy_kw_args)

            rewards.append(stats['total_reward'].mean())

        rewards = np.asarray(rewards)

        # Preserve the computed rewards for later analysis
        sorted_rewards = sorted(rewards)
        sorted_named_rewards = {}
        for i, sorted_reward in enumerate(sorted_rewards):
            reward_name = 'canidate_{}'.format(i)
            sorted_named_rewards[reward_name] = sorted_reward
        training_statistics_pre_df[generation] = sorted_named_rewards

        # Log some statistics.
        took_seconds = (dt.utcnow() - start_time).total_seconds()
        log.info('Finished generation in {:.2f} seconds.'
                  .format(took_seconds))
        log.info('Mean reward: {:.02f}'.format(rewards.mean()))
        log.info('Max reward:  {:.02f}'.format(rewards.max()))
        log.info('Reward std:  {:.02f}'.format(rewards.std()))

        log.info('Saving model of best agent and CMA state')
        best_agent_weights_flat = weights_canidates[rewards.argmax()]
        best_agent_weights = flat_weights_to_model_format(weights_slices, 
                                                          weights_shapes,
                                                          best_agent_weights_flat)
        agent_model.set_weights(best_agent_weights)
        agent_model_checkpoint_fn = ('{}/agent_model_checkpoint_generation_{:04d}.h5'
                                     .format(AGENT_MODEL_CHECKPOINTS_DIR, generation))
        agent_model.save(agent_model_checkpoint_fn)

        agent_cma_checkpoint_fn = ('{}/agent_cma_checkpoint_generation_{:04d}.pickle'
                                     .format(AGENT_CMA_CHECKPOINTS_DIR, generation))
        pd.to_pickle(es, agent_cma_checkpoint_fn)

        # Use negative rewards as this is a minimizer.
        es.tell(weights_canidates, rewards*-1)

        # Backup the rewards development for analysis.
        if generation % 1 == 0:
            training_statistics = pd.DataFrame(training_statistics_pre_df).T
            
        if generation == cma_max_generations:
            break
            training_statistics.to_pickle('{}/agent_training_statistics.pic'.format(RESULTS_DIR))

    # Save the last, hopefully the best agent model.
    agent_model.save('{}/agent_model.h5'.format(RESULTS_DIR))

    # Save the rewards development for analysis.
    training_statistics = pd.DataFrame(training_statistics_pre_df).T
    training_statistics.to_pickle('{}/agent_training_statistics.pic'.format(RESULTS_DIR))

    log.info('')
    log.info('Finished Agent Training run.')
    
    
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape=(256, 192, 3),
              kernel_size=3,
              latent_dim=32,
              first_conv_layer_filter_number=32,
              no_of_conv_layers=4,
              learning_rate=0.0001,
              kl_factor=0.00001,
              variational_mode=False,
              print_summaries=True,
              trainable_layers=[],
              shortwire_layers=[],
              **not_used_kwargs):
    """
    Build the VAE model.
    
    Adapted from:
    https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py
    
    With additional inspiration from:
    https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
    
    Arguments:
    ----------
    input_shape: tuple of int
        The shape expected as input to the VAE.
    kernel_size
        Kernel size used for convolution and deconvolution.
    latent_dim: int
        The dimension of the latent space, aka. the bottleneck.
        That is, how many float values are available to store the
        information of one frame.
    first_conv_layer_filter_number: int
        The number of filters of the first convolutional layer.
        All remaining filter counts are computed automatically.
    no_of_conv_layers: int
        How many convolutional layers to use for encoding and 
        decoding respectively.
    learning_rate: float
        Forwarded to the optimizer.
    kl_factor: float
        The weigting of the KL Regulation compared to reconstruction
        loss.
    variational_mode: bool
        If yes will create an variational auto encoder with sampling
        of the latent representation. If False will use a normal auto
        encoder. See the code for details.
    trainable_layers: list
        List of layer names that should be marked as trainable.
        If empty, everything is trainable.
    print_summaries: bool
        If true, will log summaries of the created models.
    """
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Build encoding convolution layer by layer.
    # Check for every layer if it should be trainable. 
    filters = int(first_conv_layer_filter_number / 2)
    for i in range(1, no_of_conv_layers + 1):
        filters *= 2
        layer_name = 'encoder_conv_{}'.format(i)
        if trainable_layers and layer_name not in trainable_layers:
            layer_trainable = False
        else:
            layer_trainable = True

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same',
                   name=layer_name,
                   trainable=layer_trainable)(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)
    x = Flatten()(x)

    if variational_mode:
        # Initialise with mean=1 and std=1 to prevent NN from easily 
        # finding the trival solution of setting z to zero as Space 
        # Invaders has lot's of black in the images.
        layer_name = 'z_mean'
        if trainable_layers and layer_name not in trainable_layers:
            layer_trainable = False
        else:
            layer_trainable = True
        z_mean = Dense(latent_dim,
                       name=layer_name,
                       kernel_initializer='ones',
                       trainable=layer_trainable)(x)

        layer_name = 'z_log_var'
        if trainable_layers and layer_name not in trainable_layers:
            layer_trainable = False
        else:
            layer_trainable = True
        z_log_var = Dense(latent_dim,
                          name=layer_name,
                          kernel_initializer='zeros',
                          trainable=layer_trainable)(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape"---- isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    else:
        # Normal Autoencoder bottleneck. Init as ones for the same 
        # reason pointed out above.
        layer_name = 'z'
        if trainable_layers and layer_name not in trainable_layers:
            layer_trainable = False
        else:
            layer_trainable = True
        z = Dense(latent_dim,
                  name=layer_name,
                  kernel_initializer='ones',
                  trainable=layer_trainable)(x)
        encoder = Model(inputs, z, name='encoder')

    if print_summaries:
        encoder.summary(print_fn=logging.info)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='latent_inputs')

    layer_name = 'z_inflate'
    if trainable_layers and layer_name not in trainable_layers:
        layer_trainable = False
    else:
        layer_trainable = True
    x = Dense(shape[1] * shape[2] * shape[3],
              activation='relu',
              name=layer_name,
              trainable=layer_trainable)(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(no_of_conv_layers, 0, -1):
        layer_name = 'decoder_deconv_{}'.format(i)
        if trainable_layers and layer_name not in trainable_layers:
            layer_trainable = False
        else:
            layer_trainable = True
        filters //= 2
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same',
                            name=layer_name,
                            trainable=layer_trainable)(x)

    # Flatten the feature map layers to a picture
    layer_name = 'decoder_output'
    if trainable_layers and layer_name not in trainable_layers:
        layer_trainable = False
    else:
        layer_trainable = True
    outputs = Conv2DTranspose(filters=3,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name=layer_name,
                              trainable=layer_trainable)(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    if print_summaries:
        decoder.summary(print_fn=logging.info)

    if not shortwire_layers:
        # instantiate VAE model
        if variational_mode:
            outputs = decoder(encoder(inputs)[2])
        else:
            outputs = decoder(encoder(inputs))
        vae = Model(inputs, outputs, name='vae')
        if print_summaries:
            vae.summary(print_fn=logging.info)

        # Create loss function. Use weighting of KL and MAE terms as
        # suggested by memo in:
        # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
        # The MSE here is giving a single loss value between 0 and 1.
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))

        if variational_mode:
            # This computes the KL loss for every item in the batch
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            # Take the average KL loss, only one number >= 0.
            kl_loss = K.mean(kl_loss)
            vae_loss = K.mean(reconstruction_loss + kl_factor * kl_loss)

        else:
            vae_loss = reconstruction_loss
        vae.add_loss(vae_loss)
        optimzer = RMSprop(lr=learning_rate)
        vae.compile(optimizer=optimzer)
    else:
        # Create a model that can connect intermediate layers of the vae.
        # Trough this it is possible to train only parts of the VAE.
        sw_encode_layer_name, sw_decode_layer_name = shortwire_layers
        x = inputs
        for layer in encoder.layers[1:]:
            x = layer(x)
            if layer.name == sw_encode_layer_name:
                break

        found_decoder_entry_layer = False
        for layer in decoder.layers:
            if layer.name == sw_decode_layer_name:
                found_decoder_entry_layer = True
            if found_decoder_entry_layer:
                x = layer(x)
        sw_output = x

        shortwire_model = Model(inputs, sw_output, name='shortwire_model')
        if print_summaries:
            shortwire_model.summary(print_fn=logging.info)

        sw_reconstruction_loss = mse(K.flatten(inputs), K.flatten(sw_output))
        shortwire_model.add_loss(sw_reconstruction_loss)
        optimzer = RMSprop(lr=learning_rate)
        shortwire_model.compile(optimizer=optimzer)

        vae = shortwire_model

    return encoder, decoder, vae

def run_vae_training(run_name,
                     frames_per_epoch=5000,
                     n_epochs=1000,
                     n_epochs_vae=1,
                     batch_size=128,
                     input_shape=(256, 192, 3),
                     variational_mode=False,
                     desired_train_loss=0.0001,
                     **vae_kwargs):
    """
    Executes the training of the (V)AE.
    
    This will create lot's of data into results/{}.format(run_name).
    Will also log training process to stdout and agent_training.log file.
    
    Hint: The originial WorldModels VAE is trained 10**1 epochs on 10**7
    steps of gameplay with a batch size of 100. 
    
    Arguments:
    ----------
    run_name: str
        Identifier of the current experiement, is also the folder for output.
    frames_per_epoch: int
        How many frames of gameplay should be used for training of one epoch
    n_epochs: int
        Number of epochs to train with a new set of training date.
    n_epochs_vae: int
        Number of times one epoch dataset from above is used for training
    batch_size: int
        You'll guess it.
    input_shape: tuple of int
        The shape expected as input to the VAE.
    variational_mode: bool
        see build_vae()
    desired_train_loss: float
        if Train loss of episode is lower then this value exit training.
    vae_kwargs: dict
        Given to functioncall with **{..}. Will be passed to build_vae()
        So Check the function docstring what should/could be in there.
    """
    
    # Create folders for results of training run.
    RESULTS_DIR = 'results/{}'.format(run_name)
    VAE_TEST_DIR = '{}/vae_tests'.format(RESULTS_DIR)
    VAE_ENCODER_CHECKPOINT_DIR = '{}/vae_encoder_checkpoints'.format(RESULTS_DIR)
    VAE_DECODER_CHECKPOINT_DIR = '{}/vae_decoder_checkpoints'.format(RESULTS_DIR)
    for dirname in [VAE_TEST_DIR, VAE_ENCODER_CHECKPOINT_DIR, VAE_DECODER_CHECKPOINT_DIR]:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    # Configure logger to log to notebook and file
    log = logging.getLogger('tensorflow')
    log.handlers = []
    file_handler = logging.FileHandler("{}/vae_training.log".format(RESULTS_DIR))
    stream_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(format='%(asctime)s %(levelname)s:     %(message)s ',
                    handlers=[file_handler, stream_handler],
                    level=logging.INFO)

    log.info('\n\n\n')
    log.info('Started New VAE Training run.')

    eop = EpochObsProvider(n_steps=frames_per_epoch,
                           black_bound_shape=input_shape[:2],
                           n_queued_obs=5,
                           env_name='SpaceInvaders-v4')

    encoder, decoder, vae = build_vae(input_shape=input_shape,
                                      variational_mode=variational_mode,
                                      **vae_kwargs)

    # Restore last vae training state if existing.
    # Start at the very beginning if not.
    # Extract the last checkpoint for which encoder and decoder exist.
    all_encoder_checkpoints_fns = sorted(os.listdir(VAE_ENCODER_CHECKPOINT_DIR))
    all_decoder_checkpoints_fns = sorted(os.listdir(VAE_DECODER_CHECKPOINT_DIR))

    found_last_checkpoint = False
    for encoder_checkpoint_fn in all_encoder_checkpoints_fns[::-1]:
        encoder_checkpoint_fn_no_ext = os.path.splitext(encoder_checkpoint_fn)[0]
        epoch = int(encoder_checkpoint_fn_no_ext.split('_')[-1])
        for decoder_checkpoint_fn in all_encoder_checkpoints_fns[::-1]:
            if str(epoch) in decoder_checkpoint_fn.split('_')[-1]:
                found_last_checkpoint = True
                break
        if found_last_checkpoint:
            break
    if not found_last_checkpoint:
        epoch = 0
    else: 
        encoder_checkpoint_fn = ('{}/encoder_epoch_{:04d}.h5'
                                 .format(VAE_ENCODER_CHECKPOINT_DIR, epoch))
        decoder_checkpoint_fn = ('{}/decoder_epoch_{:04d}.h5'
                                 .format(VAE_DECODER_CHECKPOINT_DIR, epoch))
        encoder_checkpoint = load_model(encoder_checkpoint_fn)
        decoder_checkpoint = load_model(decoder_checkpoint_fn)
    
        encoder.set_weights(encoder_checkpoint.get_weights())
        decoder.set_weights(decoder_checkpoint.get_weights())
        
        log.info('Restored training state of epoch {}'.format(epoch))

    # Use the existing test frame rather then creating a new one.
    test_frame_fn = '{}/test_frame.npy'.format(RESULTS_DIR)
    test_frame_existing = False
    if os.path.isfile(test_frame_fn):
        test_frame = np.load(test_frame_fn)
        test_frame_existing = True

    # Train loop. These are note the same epochs as in the VAE arg.
    # This epoch starts with a new sample of observations.
    training_statistics_pre_df = {}
    while epoch < n_epochs:
        epoch += 1
        epoch_start = dt.now()
        log.info('')
        log.info('Started Epoch {}'.format(epoch))

        # Get obs for epoch
        epoch_obs = eop.pop_observations()

        # Get a test frame for the training, but not the first one
        # as it appears to be a simple case
        if not test_frame_existing:
            test_frame = epoch_obs[100].copy()
            test_frame_fn = ('{}/test_frame'
                             .format(RESULTS_DIR))
            np.save(test_frame_fn, test_frame)

        # Make it more interesting for the VAE
        np.random.shuffle(epoch_obs)

        x = epoch_obs.astype(np.float)/255.0
        fit_res = vae.fit(x,
                          epochs=n_epochs_vae,
                          batch_size=batch_size)
        train_loss = fit_res.history['loss'][0]

        # Log measures of this epochs result.
        epoch_end = dt.now()
        epoch_walltime = (epoch_end - epoch_start).total_seconds()
        log.info('Epoch statistics:')
        log.info('Walltime:            {}'.format(epoch_walltime))
        log.info('Train loss:          {:.4f}'.format(train_loss))

        # Preserve statisics of the training
        training_statistics_pre_df[epoch] = {'train_loss': train_loss,
                                             'walltime_[s]': epoch_walltime}

        # Create a test image every now and then to 
        # visualize the progress of the vae training.
        if epoch % 2== 0:
            vae_fig_fn = ('{}/vae_test_epoch_{:04d}'
                          .format(VAE_TEST_DIR, epoch))
            save_vae_test(vae, test_frame, vae_fig_fn)

        # Backup the VAE. Save encoder and decoder individually as the
        # combined VAE model can't be restored by Keras (Bug)
            
        # Save an intermediate version of the encoder.
        if epoch % 5 == 0:
            encoder_checkpoint_fn = ('{}/encoder_epoch_{:04d}.h5'
                                     .format(VAE_ENCODER_CHECKPOINT_DIR, epoch))
            encoder.save(encoder_checkpoint_fn)
            log.info('Created encoder checkpoint {}'.format(encoder_checkpoint_fn))

        # Save an intermediate version of the decoder.
        if epoch % 5 == 0:
            decoder_checkpoint_fn = ('{}/decoder_epoch_{:04d}.h5'
                                     .format(VAE_DECODER_CHECKPOINT_DIR, epoch))
            decoder.save(decoder_checkpoint_fn)
            log.info('Created decoder checkpoint {}'.format(decoder_checkpoint_fn))
            
        # Exit trainig if loss is good.
        # Also create checkpoints as starting points for sequentiel training
        if train_loss <= desired_train_loss:
            encoder_checkpoint_fn = ('{}/encoder_epoch_{:04d}.h5'
                                     .format(VAE_ENCODER_CHECKPOINT_DIR, epoch))
            encoder.save(encoder_checkpoint_fn)
            log.info('Created encoder checkpoint {}'.format(encoder_checkpoint_fn))
            
            decoder_checkpoint_fn = ('{}/decoder_epoch_{:04d}.h5'
                                     .format(VAE_DECODER_CHECKPOINT_DIR, epoch))
            decoder.save(decoder_checkpoint_fn)
            log.info('Created decoder checkpoint {}'.format(decoder_checkpoint_fn))
            break

        # TODO Add better handling and continous backup of training statistics.


    # Save the encoder and decoder after the training. 
    # For the training of the agent and analysis.
    encoder.save('{}/encoder_model.h5'.format(RESULTS_DIR))
    decoder.save('{}/decoder_model.h5'.format(RESULTS_DIR))
    training_statistics_df = pd.DataFrame(training_statistics_pre_df).T
    training_statistics_df.to_pickle('{}/vae_training_statistics.pic'
                                     .format(RESULTS_DIR))
    log.info('')
    log.info('Finished VAE Training run.')
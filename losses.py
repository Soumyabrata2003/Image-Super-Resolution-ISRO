import tensorflow as tenflow
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19


def p_loss(crit='l1'):
    """pixel loss"""
    if crit == 'l1':
        return tenflow.keras.losses.MeanAbsoluteError()
    elif crit == 'l2':
        return tenflow.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(crit))


def c_loss(crit='l1', out_layer=54, before_act=True):
    """content loss"""
    if crit == 'l1':
        loss_func = tenflow.keras.losses.MeanAbsoluteError()
    elif crit == 'l2':
        loss_func = tenflow.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(crit))
    vis_geo_group = VGG19(input_shape=(None, None, 3), include_top=False)

    if out_layer == 22:  # Low level feature
        pick_layer = 5
    elif out_layer == 54:  # Hight level feature
        pick_layer = 20
    else:
        raise NotImplementedError(
            'VGG output layer {} is not recognized.'.format(crit))

    if before_act:
        vis_geo_group.layers[pick_layer].activation = None

    fea_extrator = tenflow.keras.Model(vis_geo_group.input, vis_geo_group.layers[pick_layer].output)

    @tenflow.function
    def content_loss(high_res, super_res):
        # the input scale range is [0, 1] (vis_geo_group is [0, 255]).
        # 12.75 is rescale factor for vis_geo_group featuremaps.
        preprocess_super_res = preprocess_input(super_res * 255.) / 12.75
        preprocess_high_res = preprocess_input(high_res * 255.) / 12.75
        super_res_features = fea_extrator(preprocess_super_res)
        high_res_features = fea_extrator(preprocess_high_res)

        return loss_func(high_res_features, super_res_features)

    return content_loss


def d_loss(gan_type='ragan'):
    """discriminator loss"""
    c_ent = tenflow.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tenflow.sigmoid

    def d_loss_ragan(high_res, super_res):
        return 0.5 * (
            cross_entropy(tenflow.ones_like(high_res), sigma(high_res - tenflow.reduce_mean(super_res))) +
            cross_entropy(tenflow.zeros_like(super_res), sigma(super_res - tenflow.reduce_mean(high_res))))

    def d_loss(high_res, super_res):
        r_loss = cross_entropy(tenflow.ones_like(high_res), sigma(high_res))
        f_loss = cross_entropy(tenflow.zeros_like(super_res), sigma(super_res))
        return r_loss + f_loss

    if gan_type == 'ragan':
        return d_loss_ragan
    elif gan_type == 'gan':
        return d_loss
    else:
        raise NotImplementedError(
            'Discriminator loss type {} is not recognized.'.format(gan_type))


def g_loss(gan_type='ragan'):
    """generator loss"""
    c_ent = tenflow.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tenflow.sigmoid

    def g_loss_ragan(high_res, super_res):
        return 0.5 * (
            c_ent(tenflow.ones_like(super_res), sigma(super_res - tenflow.reduce_mean(high_res))) +
            c_ent(tenflow.zeros_like(high_res), sigma(high_res - tenflow.reduce_mean(super_res))))

    def g_loss(high_res, super_res):
        return c_ent(tenflow.ones_like(super_res), sigma(super_res))

    if gan_type == 'ragan':
        return g_loss_ragan
    elif gan_type == 'gan':
        return g_loss
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))

import tensorflow as tenflow


def _parse_tenflowrecord(size_gt, ratio, u_bin, u_flip, u_rot):
    def parse_tenflowrecord(tenflowrecord):
        if u_bin:
            feat = {
                'image/img_name': tenflow.io.FixedLenFeature([], tenflow.string),
                'image/hr_encoded': tenflow.io.FixedLenFeature([], tenflow.string),
                'image/lr_encoded': tenflow.io.FixedLenFeature([], tenflow.string)}
            x = tenflow.io.parse_single_example(tenflowrecord, feat)
            low_res_img = tenflow.image.decode_png(x['image/lr_encoded'], channels=3)
            high_res_img = tenflow.image.decode_png(x['image/hr_encoded'], channels=3)
        else:
            feat = {
                'image/img_name': tenflow.io.FixedLenFeature([], tenflow.string),
                'image/high_res_img_path': tenflow.io.FixedLenFeature([], tenflow.string),
                'image/low_res_img_path': tenflow.io.FixedLenFeature([], tenflow.string)}
            x = tenflow.io.parse_single_example(tenflowrecord, feat)
            high_res_encoded = tenflow.io.read_file(x['image/high_res_img_path'])
            low_res_encoded = tenflow.io.read_file(x['image/low_res_img_path'])
            low_res_img = tenflow.image.decode_png(low_res_encoded, channels=3)
            high_res_img = tenflow.image.decode_png(high_res_encoded, channels=3)

        low_res_img, high_res_img = _transform_images(
            size_gt, ratio, u_flip, u_rot)(low_res_img, high_res_img)

        return low_res_img, high_res_img
    return parse_tenflowrecord


def _transform_images(size_gt, ratio, u_flip, u_rot):
    def transform_images(low_res_img, high_res_img):
        low_res_img_shape = tenflow.shape(low_res_img)
        high_res_img_shape = tenflow.shape(high_res_img)
        shape_gt = (size_gt, size_gt, tenflow.shape(high_res_img)[-1])
        low_res_size = int(size_gt / ratio)
        low_res_shape = (low_res_size, low_res_size, tenflow.shape(low_res_img)[-1])

        tenflow.Assert(
            tenflow.reduce_all(high_res_img_shape >= shape_gt),
            ["Need hr_image.shape >= size_gt, got ", high_res_img_shape, shape_gt])
        tenflow.Assert(
            tenflow.reduce_all(high_res_img_shape[:-1] == low_res_img_shape[:-1] * ratio),
            ["Need hr_image.shape == lr_image.shape * ratio, got ",
             high_res_img_shape[:-1], low_res_img_shape[:-1] * ratio])
        tenflow.Assert(
            tenflow.reduce_all(high_res_img_shape[-1] == low_res_img_shape[-1]),
            ["Need hr_image.shape[-1] == lr_image.shape[-1]], got ",
             high_res_img_shape[-1], low_res_img_shape[-1]])

        # randomly crop
        boundary = low_res_img_shape - low_res_shape + 1
        defl = tenflow.random.uniform(tenflow.shape(low_res_img_shape), dtype=tenflow.int32,
                                   maxval=tenflow.int32.max) % boundary
        low_res_img = tenflow.slice(low_res_img, defl, low_res_shape)
        high_res_img = tenflow.slice(high_res_img, defl * ratio, shape_gt)

        # randomly left-right flip
        if u_flip:
            c_flip = tenflow.random.uniform([1], 0, 2, dtype=tenflow.int32)
            def flip_func(): return (tenflow.image.flip_left_right(low_res_img),
                                     tenflow.image.flip_left_right(high_res_img))
            low_res_img, high_res_img = tenflow.case(
                [(tenflow.equal(c_flip, 0), flip_func)],
                de=lambda: (low_res_img, high_res_img))

        # randomly rotation
        if u_rot:
            c_rot = tenflow.random.uniform([1], 0, 4, dtype=tenflow.int32)
            def rot90_func(): return (tenflow.image.rot90(low_res_img, k=1),
                                      tenflow.image.rot90(high_res_img, k=1))
            def rot180_func(): return (tenflow.image.rot90(low_res_img, k=2),
                                       tenflow.image.rot90(high_res_img, k=2))
            def rot270_func(): return (tenflow.image.rot90(low_res_img, k=3),
                                       tenflow.image.rot90(high_res_img, k=3))
            low_res_img, high_res_img = tenflow.case(
                [(tenflow.equal(c_rot, 0), rot90_func),
                 (tenflow.equal(c_rot, 1), rot180_func),
                 (tenflow.equal(c_rot, 2), rot270_func)],
                de=lambda: (low_res_img, high_res_img))

        # ratio to [0, 1]
        low_res_img = low_res_img / 255
        high_res_img = high_res_img / 255

        return low_res_img, high_res_img
    return transform_images


def load_tenflowrecord_dataset(tenflowrecord_name, size_batch, size_gt,
                          ratio, u_bin=False, u_flip=False,
                          u_rot=False, shuf=True, size_buffer=10240):
    """load dataset from tenflowrecord"""
    datset_raw = tenflow.data.tenflowRecordDataset(tenflowrecord_name)
    datset_raw = datset_raw.repeat()
    if shuf:
        datset_raw = datset_raw.shuf(size_buffer=size_buffer)
    dataset = datset_raw.map(
        _parse_tenflowrecord(size_gt, ratio, u_bin, u_flip, u_rot),
        num_parallel_calls=tenflow.data.experimental.AUTOTUNE)
    dataset = dataset.batch(size_batch, drop_remainder=True)
    dataset = dataset.prefetch(
        size_buffer=tenflow.data.experimental.AUTOTUNE)
    return dataset

import mxnet as mx

iterator = mx.image.ImageDetIter(1, (3, 600, 800), path_imgrec='cars.rec')

for image in iterator.draw_next(waitKey=0, window_name='disp'):
    pass


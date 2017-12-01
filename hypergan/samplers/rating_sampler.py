from hypergan.samplers.base_sampler import BaseSampler
import tensorflow as tf
import numpy as np

class RatingSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None
        self.d_real = None

    def _sample(self):
        gan = self.gan
        z_t = gan.encoder.z
        inputs_t = gan.inputs.x

        if self.z is None:
            self.z = gan.encoder.z.eval()
            self.input = gan.session.run(gan.inputs.x)
            self.d_real, _ = gan.split_batch(gan.discriminator.sample)

        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            d_real, generator = gan.session.run((self.d_real, gan.generator.sample), feed_dict={z_t: self.z, inputs_t: self.input})
            np.savetxt('samples/rating.csv', d_real)
            return {'generator': generator}
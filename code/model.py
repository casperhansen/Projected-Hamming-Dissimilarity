import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import numpy as np
from tensorflow.losses import compute_weighted_loss, Reduction


def hinge_loss_eps(labels, logits, epsval, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.to_float(logits)
    labels = math_ops.to_float(labels)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_eps = array_ops.ones_like(labels)*epsval
    all_ones = array_ops.ones_like(labels)

    labels = math_ops.subtract(2 * labels, all_ones)
    losses = nn_ops.relu(
        math_ops.subtract(all_eps, math_ops.multiply(labels, logits)))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)

class Model():
    def __init__(self, sample, args):
        self.sample = sample
        self.batchsize = args["batchsize"]

    def _make_embedding(self, vocab_size, embedding_size, name):
        W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1, maxval=1),
                        trainable=True, name=name)

        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        embedding_init = W.assign(embedding_placeholder)
        return (W, embedding_placeholder, embedding_init)

    def _extract(self, item_emb, user_embs, user_mask_emb):#, user_index_embedding_matrix, user_index_embedding_ratings_matrix):
        user, i1, i2, iu, i1r, i2r = self.sample[0], self.sample[1], self.sample[2], self.sample[3], self.sample[4], self.sample[5]

        user_mask = tf.nn.embedding_lookup(user_mask_emb, user)

        #user_index_values = tf.cast(tf.nn.embedding_lookup(user_index_embedding_matrix, user), tf.int32)
        #user_index_ratings_values = tf.nn.embedding_lookup(user_index_embedding_ratings_matrix, user)

        user = tf.nn.embedding_lookup(user_embs, user) #tf.concat([tf.expand_dims(tf.nn.embedding_lookup(user_emb_x, user), axis=-1) for user_emb_x in user_embs], axis=-1)

        i1 = tf.nn.embedding_lookup(item_emb, i1)
        i2 = tf.nn.embedding_lookup(item_emb, i2)
        i3 = tf.nn.embedding_lookup(item_emb, iu)

        return user, user_mask, i1, i2, i3, i1r, i2r#, user_index_values, user_index_ratings_values

    def _sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self._sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature, axis=-1)

    def gumbel_softmax(self, logits, temperature, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, bits, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, bits, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self._gumbel_softmax_sample(logits, temperature)
        #if hard:

        # k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keepdims=True)), y.dtype)
        #print(tf.reduce_max(y, -1), y)
        #exit()
        y_hard = tf.stop_gradient(y_hard - y) + y

        y = tf.cond(hard, lambda: y_hard, lambda: y)
        return y

    def make_network(self, item_emb, user_embs, user_mask_emb, is_training, args, max_rating, sigma_anneal, batchsize):
        # user_index_embedding_matrix, user_index_embedding_ratings_matrix,

        #################### Bernoulli Sample #####################
        ## ref code: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        def bernoulliSample(x):
            """
            Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
            using the straight through estimator for the gradient.
            E.g.,:
            if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
            and the gradient will be pass-through (identity).
            """
            g = tf.get_default_graph()

            with ops.name_scope("BernoulliSample") as name:
                with g.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):

                    if args["deterministic_train"]:
                        train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)
                    else:
                        train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))

                    if args["deterministic_eval"]:
                        eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)
                    else:
                        eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))

                    mus = tf.cond(is_training, train_fn, eval_fn)

                    return tf.ceil(x - mus, name=name)

        @ops.RegisterGradient("BernoulliSample_ST")
        def bernoulliSample_ST(op, grad):
            return [grad, tf.zeros(tf.shape(op.inputs[1]))]

        ###########################################################

        user, user_mask, i1, i2, i3, i1r, i2r = self._extract(item_emb, user_embs, user_mask_emb)

        user = tf.sigmoid(user)#[:, :, 0])

        i1 = tf.sigmoid(i1)
        i2 = tf.sigmoid(i2)

        i1_sampling = i1
        i2_sampling = i2
        user_sampling = user


        user = bernoulliSample(user)

        if args["usermask_nograd"]:
            selfmask = tf.stop_gradient(user)
        else:
            selfmask = user

        i1_org = bernoulliSample(i1) * (selfmask if args["optimize_selfmask"] else 1)
        i2_org = bernoulliSample(i2) * (selfmask if args["optimize_selfmask"] else 1)

        user_m = 2*user - 1
        i1_org_m = (2*i1_org - 1) #* tf.cast(tf.abs(user) > 0.5, tf.float32) #* ((user+1)/2)
        i2_org_m = (2*i2_org - 1) #* tf.cast(tf.abs(user) > 0.5, tf.float32) #* ((user+1)/2)

        nonzero_bits = args["bits"]# tf.reduce_sum(tf.cast(user_m>0.5, tf.float32), axis=-1)

        def make_total_loss(i1_org, i2_org, i1r, i2r, i1_sampling, i2_sampling, anneal):
            e0 = tf.random.normal([batchsize], stddev=1.0,
                                     name='normaldis0')
            e1 = tf.truncated_normal([batchsize, args["bits"]], stddev=1.0,
                                     name='normaldist1')  # tf.random.normal([batchsize, args["bits"]])
            e2 = tf.truncated_normal([batchsize, args["bits"]], stddev=1.0, name='normaldist2')

            i1 = i1_org# e1*0 + i1_org
            i2 = i2_org#e2*0 + i2_org

            i1r = i1r + e0*anneal

            #i1r_m = 2*args["bits"] * (i1r/max_rating) - args["bits"]
            i1r_m = 2*nonzero_bits * (i1r/max_rating) - nonzero_bits

            dot_i1 = tf.reduce_sum(user_m * i1, axis=-1) #- tf.cond(is_training, lambda :(1.0*args["bits"] - nonzero_bits), lambda :tf.reduce_sum(tf.cast(user_m<0.5, tf.float32), axis=-1))
            dot_i2 = tf.reduce_sum(user_m * i2, axis=-1) #- tf.cond(is_training, lambda :(1.0*args["bits"] - nonzero_bits), lambda :tf.reduce_sum(tf.cast(user_m<0.5, tf.float32), axis=-1))

            loss = tf.reduce_mean(tf.math.pow(i1r_m - dot_i1, 2), axis=-1)

            # ranking loss
            different_rating = tf.cast(tf.abs(i1r - i2r) > 0.0001, tf.float32)
            same_rating = tf.cast(different_rating < 0.5, tf.float32)

            signpart = tf.cast(i1r > i2r, tf.float32)
            rank_loss_uneq = hinge_loss_eps(labels=signpart, logits=(dot_i1 - dot_i2), epsval=1.0, weights=different_rating)
            eq_dist = tf.math.pow(dot_i1 - dot_i2, 2)
            rank_loss_eq = compute_weighted_loss(eq_dist, weights=same_rating, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

            # reg
            loss_kl = tf.multiply(i1_sampling, tf.math.log(tf.maximum(i1_sampling / 0.5, 1e-10))) + \
                      tf.multiply(1 - i1_sampling, tf.math.log(tf.maximum((1 - i1_sampling) / 0.5, 1e-10)))

            loss_kl = tf.reduce_sum(tf.reduce_sum(loss_kl, 1) / args["batchsize"], axis=0)

            loss_kl_user = tf.multiply(user_sampling, tf.math.log(tf.maximum(user_sampling / 0.5, 1e-10))) + \
                      tf.multiply(1 - user_sampling, tf.math.log(tf.maximum((1 - user_sampling) / 0.5, 1e-10)))

            loss_kl_user = tf.reduce_sum(tf.reduce_sum(loss_kl_user, 1) / args["batchsize"], axis=0)

            # combine losses
            total_loss = loss + args["KLweight"]*(loss_kl + loss_kl_user) #+ (1-anneal)*(rank_loss_uneq + rank_loss_eq) # loss    loss_kl*0.1*(1-anneal)
            ##total_loss = loss + (1-anneal)*(rank_loss_uneq + rank_loss_eq) # loss    loss_kl*0.1*(1-anneal)

            i1_dist = -dot_i1
            return total_loss, i1_dist, loss, rank_loss_uneq, rank_loss_eq, i1r_m

        total_loss, ham_dist_i1, reconloss, rank_loss_uneq, rank_loss_eq, i1r_m = make_total_loss(i1_org_m, i2_org_m, i1r, i2r, i1_sampling, i2_sampling, sigma_anneal)

        total = total_loss

        if args["force_selfmask"]:
            i1_org_m = (i1_org_m + 1) / 2
            i1_org_m = i1_org_m * selfmask
            i1_org_m = 2*i1_org_m - 1

        return total, total, ham_dist_i1, i1_org_m, user_m, reconloss, rank_loss_uneq, rank_loss_eq, i1r_m,tf.reduce_sum(tf.cast(user_m>0.5, tf.float32), axis=-1)



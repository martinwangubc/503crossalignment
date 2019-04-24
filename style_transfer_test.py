import os
import sys
import time
import ipdb
import random
import cPickle as pickle
import numpy as np
import tensorflow as tf

from vocab import Vocabulary, build_vocab
from accumulator import Accumulator
from options3 import load_arguments
from file_io import load_sent, write_sent
from utils import *
from nn import *
import beam_search, greedy_decoding

class Model(object):

    def __init__(self, args, vocab):
        #tf.get_variable_scope().reuse_variables()
        dim_y = args.dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        beta1, beta2 = 0.5, 0.999
        grad_clip = 30.0

        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.rho = tf.placeholder(tf.float32,
            name='rho')
        self.gamma = tf.placeholder(tf.float32,
            name='gamma')

        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None],    #size * len
            name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')
        self.labels = tf.placeholder(tf.float32, [None],
            name='labels')

        # testing optimization
        testing1 = tf.constant([[37.0, -23.0], [1.0, 4.0]])
        testing2 = tf.constant([[37.0, -23.0], [1.0, 4.0]])
        self.lineartest = tf.matmul(testing1,testing2)
        #=====

        labels = tf.reshape(self.labels, [-1, 1])

        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))

        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####
        init_state = tf.concat([linear(labels, dim_y, scope='encoder'),
            tf.zeros([self.batch_size, dim_z])], 1)
        cell_e = create_cell(dim_h, n_layers, self.dropout)
        _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
            initial_state=init_state, scope='encoder')
        z = z[:, dim_y:]

        #cell_e = create_cell(dim_z, n_layers, self.dropout)
        #_, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
        #    dtype=tf.float32, scope='encoder')

        self.h_ori = tf.concat([linear(labels, dim_y,
            scope='generator'), z], 1)
        self.h_tsf = tf.concat([linear(1-labels, dim_y,
            scope='generator', reuse=True), z], 1)

        cell_g = create_cell(dim_h, n_layers, self.dropout)
        g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
            initial_state=self.h_ori, scope='generator')

        #======
        # creating new decoder modules here =====

        #NEW PLACEHOLDER VARIABLES
        self.testing = tf.placeholder(tf.float32,name = 'testing')

        # CURRENTLY it replicates the functitonality of the first one. need to
        # modify the inputs (placeeholders) in the tensorflow graph accordingly.

        # z is shared (encoder shared), output passes to second decoder pairing.
        # here, scope is "generator2"
        self.h_ori2 = tf.concat([linear(labels, dim_y,
            scope='generator2'), z], 1)
        self.h_tsf2 = tf.concat([linear(1-labels, dim_y,
            scope='generator2', reuse=True), z], 1)

        cell_g2 = create_cell(dim_h, n_layers, self.dropout)
        g_outputs2, _ = tf.nn.dynamic_rnn(cell_g2, dec_inputs,
            initial_state=self.h_ori2, scope='generator2')

        teach_h2 = tf.concat([tf.expand_dims(self.h_ori2, 1), g_outputs2], 1)
        g_outputs2 = tf.nn.dropout(g_outputs2, self.dropout)
        g_outputs2 = tf.reshape(g_outputs2, [-1, dim_h])
        g_logits2 = tf.matmul(g_outputs2, proj_W) + proj_b # change projections?

        loss_rec2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits2)
        loss_rec2 *= tf.reshape(self.weights, [-1])
        self.loss_rec2 = tf.reduce_sum(loss_rec2) / tf.to_float(self.batch_size)
        # continuing
        go = dec_inputs[:,0,:] # unchanged
        soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding,
            self.gamma)
        hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)

        soft_h_ori2, soft_logits_ori2 = rnn_decode(self.h_ori2, go, max_len,
            cell_g2, soft_func, scope='generator2')
        soft_h_tsf2, soft_logits_tsf2 = rnn_decode(self.h_tsf2, go, max_len,
            cell_g2, soft_func, scope='generator2')

        hard_h_ori2, self.hard_logits_ori2 = rnn_decode(self.h_ori2, go, max_len,
            cell_g2, hard_func, scope='generator2')
        hard_h_tsf2, self.hard_logits_tsf2 = rnn_decode(self.h_tsf2, go, max_len,
            cell_g2, hard_func, scope='generator2')

        half = self.batch_size / 2
        zeros, ones = self.labels[:half], self.labels[half:]
        soft_h_tsf2 = soft_h_tsf2[:, :1+self.batch_len, :]

        self.loss_d02, loss_g02 = discriminator(teach_h2[:half], soft_h_tsf2[half:],
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator02')
        self.loss_d12, loss_g12 = discriminator(teach_h2[half:], soft_h_tsf2[:half],
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator12')

        #####   optimizer   #####
        self.loss_adv2 = loss_g02 + loss_g12
        self.loss2 = self.loss_rec2 + self.rho * self.loss_adv2

        theta_eg2 = retrive_var(['encoder', 'generator2',
            'embedding', 'projection'])
        theta_d02 = retrive_var(['discriminator02'])
        theta_d12 = retrive_var(['discriminator12'])

        opt2 = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)

        grad_rec2, _ = zip(*opt2.compute_gradients(self.loss_rec2, theta_eg2))
        grad_adv2, _ = zip(*opt2.compute_gradients(self.loss_adv2, theta_eg2))
        grad2, _ = zip(*opt2.compute_gradients(self.loss2, theta_eg2))
        grad2, _ = tf.clip_by_global_norm(grad2, grad_clip) # grad_clip doesn't need 2

        self.grad_rec_norm2 = tf.global_norm(grad_rec2)
        self.grad_adv_norm2 = tf.global_norm(grad_adv2)
        self.grad_norm2 = tf.global_norm(grad2)

        self.optimize_tot2 = opt2.apply_gradients(zip(grad2, theta_eg2))
        self.optimize_rec2 = opt2.minimize(self.loss_rec2, var_list=theta_eg2)
        self.optimize_d02 = opt2.minimize(self.loss_d02, var_list=theta_d02)
        self.optimize_d12 = opt2.minimize(self.loss_d12, var_list=theta_d12)

        self.saver2 = tf.train.Saver()
        #======
        #======

        # attach h0 in the front
        teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        g_outputs = tf.reshape(g_outputs, [-1, dim_h])
        g_logits = tf.matmul(g_outputs, proj_W) + proj_b

        loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits)
        loss_rec *= tf.reshape(self.weights, [-1])
        self.loss_rec = tf.reduce_sum(loss_rec) / tf.to_float(self.batch_size)

        #####   feed-previous decoding   #####
        go = dec_inputs[:,0,:]
        soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding,
            self.gamma)
        hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)

        soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len,
            cell_g, soft_func, scope='generator')
        soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            cell_g, soft_func, scope='generator')

        hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len,
            cell_g, hard_func, scope='generator')
        hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            cell_g, hard_func, scope='generator')

        #####   discriminator   #####
        # a batch's first half consists of sentences of one style,
        # and second half of the other
        half = self.batch_size / 2
        zeros, ones = self.labels[:half], self.labels[half:]
        soft_h_tsf = soft_h_tsf[:, :1+self.batch_len, :]

        self.loss_d0, loss_g0 = discriminator(teach_h[:half], soft_h_tsf[half:],
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator0')
        self.loss_d1, loss_g1 = discriminator(teach_h[half:], soft_h_tsf[:half],
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator1')

        #####   optimizer   #####
        self.loss_adv = loss_g0 + loss_g1
        self.loss = self.loss_rec + self.rho * self.loss_adv

        theta_eg = retrive_var(['encoder', 'generator',
            'embedding', 'projection'])
        theta_d0 = retrive_var(['discriminator0'])
        theta_d1 = retrive_var(['discriminator1'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)

        grad_rec, _ = zip(*opt.compute_gradients(self.loss_rec, theta_eg))
        grad_adv, _ = zip(*opt.compute_gradients(self.loss_adv, theta_eg))
        grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
        grad, _ = tf.clip_by_global_norm(grad, grad_clip)

        self.grad_rec_norm = tf.global_norm(grad_rec)
        self.grad_adv_norm = tf.global_norm(grad_adv)
        self.grad_norm = tf.global_norm(grad)

        self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))
        self.optimize_rec = opt.minimize(self.loss_rec, var_list=theta_eg)
        self.optimize_d0 = opt.minimize(self.loss_d0, var_list=theta_d0)
        self.optimize_d1 = opt.minimize(self.loss_d1, var_list=theta_d1)

        self.saver = tf.train.Saver()
        # tf.get_variable_scope().reuse_variables()

def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1,
        vocab.word2id, args.batch_size)

    #data0_rec, data1_rec = [], []
    data0_tsf, data1_tsf = [], []
    losses = Accumulator(len(batches), ['loss', 'rec', 'adv', 'd0', 'd1'])
    for batch in batches:
        rec, tsf = decoder.rewrite(batch)
        half = batch['size'] / 2
        #data0_rec += rec[:half]
        #data1_rec += rec[half:]
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]

        loss, loss_rec, loss_adv, loss_d0, loss_d1 = sess.run([model.loss,
            model.loss_rec, model.loss_adv, model.loss_d0, model.loss_d1],
            feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])

    n0, n1 = len(data0), len(data1)
    #data0_rec = reorder(order0, data0_rec)[:n0]
    #data1_rec = reorder(order1, data1_rec)[:n1]
    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]

    if out_path:
        #write_sent(data0_rec, out_path+'.0'+'.rec')
        #write_sent(data1_rec, out_path+'.1'+'.rec')
        write_sent(data0_tsf, out_path+'.0'+'.tsf')
        write_sent(data1_tsf, out_path+'.1'+'.tsf')

    return losses

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model

if __name__ == '__main__':
    args = load_arguments()

    #####   data preparation   #####
    if args.train:

        # 0 is the starting style !
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        train2 = load_sent(args.train + '.2', args.max_train_size)
        print '#sents of training file 0:', len(train0)
        print '#sents of training file 1:', len(train1)
        print '#sents of training file 2:', len(train2)

        # loaded all three datasets here. Train once with 0-1 and once with 0-2

        print("=====got here training=====")

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)
        if not os.path.isfile(args.vocab2):
            build_vocab(train0 + train2, args.vocab2)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print 'vocabulary 1 size:', vocab.size

        # introduce a second input argument, "vocab2"

    vocab2 = Vocabulary(args.vocab2, args.embedding, args.dim_emb)
    print 'vocabulary 2 size:', vocab2.size

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')
        dev2 = load_sent(args.dev + '.2')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')
        test2 = load_sent(args.test + '.2')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # same session for all of it
    with tf.Session(config=config) as sess:
        print("=====entered tensorflow session=====")
        model = create_model(sess, args, vocab)
        print("=====created model=====")

        if args.beam > 1:
            decoder = beam_search.Decoder(sess, args, vocab, model)
            #decoder2 = beam_search.Decoder(sess, args, vocab2, model)
        else:
            decoder = greedy_decoding.Decoder(sess, args, vocab, model)
            #decoder2 = greedy_decoding.Decoder(sess, args, vocab2, model)

        if args.train:
            batches1, _, _ = get_batches(train0, train1, vocab.word2id,
                args.batch_size, noisy=True)
            random.shuffle(batches1)

            batches2, _, _ = get_batches(train0, train2, vocab2.word2id,
                args.batch_size, noisy=True)
            random.shuffle(batches2)

            start_time = time.time()
            step = 0
            losses = Accumulator(args.steps_per_checkpoint,
                ['loss', 'rec', 'adv', 'd0', 'd1'])
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob

            for epoch in range(1, 1+args.max_epochs):
                print '--------------------epoch %d--------------------' % epoch
                print 'learning_rate:', learning_rate, '  gamma:', gamma

                for batch in batches1:
                    feed_dict = feed_dictionary(model, batch, rho, gamma,
                        dropout, learning_rate)

                    loss_d0, _, linearoutput = sess.run([model.loss_d0, model.optimize_d0, model.lineartest],
                        feed_dict=feed_dict)
                    # if epoch == 1:
		    	    # print(linearoutput)
                    loss_d1, _ = sess.run([model.loss_d1, model.optimize_d1],
                        feed_dict=feed_dict)

                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d0 < 1.2 and loss_d1 < 1.2:
                        optimize = model.optimize_tot
                    else:
                        optimize = model.optimize_rec

                    loss, loss_rec, loss_adv, _ = sess.run([model.loss,
                        model.loss_rec, model.loss_adv, optimize],
                        feed_dict=feed_dict)
                    losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])


                    step += 1
                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,'
                            % (step, time.time() - start_time))
                        losses.clear()
                if args.dev:
                    dev_losses = transfer(model, decoder, sess, args, vocab,
                        dev0, dev1, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.values[0] < best_dev:
                        best_dev = dev_losses.values[0]
                        print 'saving model...'
                        model.saver.save(sess, args.model)

                gamma = max(args.gamma_min, gamma * args.gamma_decay)

                # SECOND HALF OF THE TRAINING

                print("started the second half of this epoch")
                for batch in batches2:
                    feed_dict = feed_dictionary(model, batch, rho, gamma,
                        dropout, learning_rate)

                    loss_d02, _, linearoutput = sess.run([model.loss_d02, model.optimize_d02, model.lineartest],
                        feed_dict=feed_dict)
                #     if epoch == 1:
                # print(linearoutput)
                    loss_d12, _ = sess.run([model.loss_d12, model.optimize_d12],
                        feed_dict=feed_dict)

                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d02 < 1.2 and loss_d12 < 1.2:
                        optimize = model.optimize_tot2
                    else:
                        optimize = model.optimize_rec2

                    loss, loss_rec, loss_adv, _ = sess.run([model.loss2,
                        model.loss_rec2, model.loss_adv2, optimize],
                        feed_dict=feed_dict)
                    losses.add([loss, loss_rec, loss_adv, loss_d02, loss_d12])


                    step += 1
                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,'
                            % (step, time.time() - start_time))
                        losses.clear()
                if args.dev:
                    dev_losses = transfer(model, decoder, sess, args, vocab2,
                        dev0, dev2, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.values[0] < best_dev:
                        best_dev = dev_losses.values[0]
                        print 'saving model...'
                        model.saver.save(sess, args.model)

                gamma = max(args.gamma_min, gamma * args.gamma_decay)
            print("ended epoch")
            # for epoch in range(1, 1+args.max_epochs):
            #     print '--------------------epoch %d--------------------' % epoch
            #     print 'learning_rate:', learning_rate, '  gamma:', gamma
            #
            #
            # print("=====ended second group of epochs=====")
    # going through twice

        if args.test:
            test_losses = transfer(model, decoder, sess, args, vocab,
                test0, test1, args.output)
            test_losses.output('test')
            test_losses2 = transfer(model, decoder, sess, args, vocab2,
                test0, test2, args.output2)
            test_losses2.output('test2')

        if args.online_testing:
            while True:
                sys.stdout.write('> ')
                sys.stdout.flush()
                inp = sys.stdin.readline().rstrip()
                if inp == 'quit' or inp == 'exit':
                    break
                inp = inp.split()
                y = int(inp[0])
                sent = inp[1:]

                batch = get_batch([sent], [y], vocab.word2id)
                ori, tsf = decoder.rewrite(batch)
                print 'original:', ' '.join(w for w in ori[0])
                print 'transfer:', ' '.join(w for w in tsf[0])
        print("=====ended main code=====")

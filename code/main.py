import argparse
import tensorflow as tf
import pickle
import glob
from nn_helpers import generator
from model import Model
import numpy as np
import time
from helpers import ndcg_score, mrr
from scipy.io import loadmat
import datetime
from bit_inspecter import bit_histogram
import multiprocessing as mp


def fprint(output_file, text):
    with open(output_file, "a") as myfile:
        myfile.write(str(text) + "\n")

def onepass(sess, eval_list, some_handle, num_samples, eval_batchsize, handle, anneal_val, batch_placeholder, is_training, force_selfmask, args):
    losses_val = []
    losses_val_recon = []
    losses_val_eq = []
    losses_val_uneq = []

    valcounter = 0
    val_user_items = {}
    total = num_samples
    valdone = False

    user_vectors = []
    item_vectors = []

    while not valdone:
        lossval, hamdist, userval, item_ratingval, itemsample, item_emb, user_emb, \
            lossval_recon, lossval_uneq, lossval_eq = sess.run(eval_list, feed_dict={handle: some_handle, is_training: False,
                                                                        anneal_val: 0,
                                                                        batch_placeholder: min(total, eval_batchsize)})

        losses_val.append(lossval)

        losses_val_recon.append(lossval_recon)
        losses_val_uneq.append(lossval_uneq)
        losses_val_eq.append(lossval_eq)

        valcounter += 1
        total -= len(userval)
        user_vectors += user_emb.tolist()
        item_vectors += item_emb.tolist()

        #print(hamdist)
        if total <= 0:
            valdone = True

        for kk in range(len(userval)):
            user = userval[kk]
            item_rating = item_ratingval[kk]

            user_item_score = -np.sum(user_emb[kk] * item_emb[kk])

            if user not in val_user_items:
                val_user_items[user] = [[], []]

            val_user_items[user][0].append(int(user_item_score))
            val_user_items[user][1].append(int(item_rating))

    assert(total == 0)
    t = 0


    inps = []
    for user in val_user_items:
        t += len(val_user_items[user][1])
        inps.append([val_user_items[user][1], val_user_items[user][0]])

    res = pool.starmap_async(ndcg_score, inps)
    ndcgs = res.get()

    res_mrr = pool.starmap_async(mrr, inps)
    mrrs = res_mrr.get()

    if not args["realvalued"]:
        user_vectors = np.array(user_vectors).astype(int)
        item_vectors = np.array(item_vectors).astype(int)
    else:
        user_vectors = np.array(user_vectors)
        item_vectors = np.array(item_vectors)

    print(np.mean(bit_histogram(user_vectors, 1)), np.mean(bit_histogram(item_vectors, 1)))
    print(user_vectors[10][:5], item_vectors[10][:5])
    print(np.unique(user_vectors.astype(int)[:]), np.unique(item_vectors.astype(int)[:]), len(item_vectors[0]), len(user_vectors[0]) )

    return np.mean(losses_val), np.mean(ndcgs, 0), np.mean(losses_val_recon), \
           np.mean(losses_val_uneq), np.mean(losses_val_eq), len(ndcgs), ndcgs, np.mean(mrrs, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", default=400, type=int) # 400 works well, but you can also search among {100, 200, 400, 800}
    parser.add_argument("--bits", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float) # 0.001 works well, but you can also search among {0.005, 0.001, 0.0005}
    parser.add_argument("--dataset", default="ml-1m", type=str) # ml-1m, ml-10m, yelp16, amazon
    parser.add_argument("--ofile", default="../output.txt", type=str) # just a log file - results are saved to a file in results/

    # just keep these fixed
    parser.add_argument("--decay_rate", default=1.0, type=float)
    parser.add_argument("--mul", default=0.5, type=float)
    parser.add_argument("--anneal_val", default=1.0, type=float)
    parser.add_argument("--deterministic_eval", default=1, type=int)
    parser.add_argument("--deterministic_train", default=0, type=int)
    parser.add_argument("--optimize_selfmask", default=1, type=int)
    parser.add_argument("--usermask_nograd", default=0, type=int)
    parser.add_argument("--KLweight", default=0.00, type=float)
    parser.add_argument("--force_selfmask", default=0, type=int)
    parser.add_argument("--realvalued", default=0, type=int)


    eval_batchsize = 2000


    args = parser.parse_args()

    savename = "results/" + "_".join([str(v) for v in [args.dataset, args.bits, args.batchsize, args.lr, args.anneal_val, args.deterministic_eval,
                                          args.deterministic_train, args.optimize_selfmask, args.usermask_nograd, args.KLweight, args.force_selfmask, args.mul]]) + "_res.pkl"

    args.realvalued = args.realvalued > 0.5
    args.deterministic_eval = args.deterministic_eval > 0.5
    args.usermask_nograd = args.usermask_nograd > 0.5
    args.deterministic_train = args.deterministic_train > 0.5
    args.optimize_selfmask = args.optimize_selfmask > 0.5
    args.force_selfmask = args.force_selfmask > 0.5

    args = vars(args)
    print(args)
    fprint(args["ofile"], args)

    basepath = "../data/"+args["dataset"]+"/tfrecord/"
    dicfile = basepath + "dict.pkl"
    dicfile = pickle.load(open(dicfile, "rb"))
    num_users, num_items = dicfile[0], dicfile[1]

    args["num_users"] = num_users
    args["num_items"] = num_items

    trainfiles = glob.glob(basepath + "*train_*tfrecord")
    valfiles = glob.glob(basepath + "*val_*tfrecord")
    testfiles = glob.glob(basepath + "*test_*tfrecord")

    # you can get the specific numbers below from the commeted-out code in (*) on line 196

    if args["dataset"].lower() == "ml-10m":
        train_samples = 4201261 
        val_samples = 779388 
        test_samples = 5014822 
        max_rating = 10.0
    elif args["dataset"].lower() == "ml-1m":
        train_samples = 420336 
        val_samples = 77436
        test_samples =  500767
        max_rating = 5.0
    elif args["dataset"].lower() == "yelp16":
        train_samples = 241876 
        val_samples = 54245
        test_samples = 306396
        max_rating = 5.0
    elif args["dataset"].lower() == "amazon":
        train_samples = 1896886
        val_samples = 417192
        test_samples = 2387890
        max_rating = 5.0
    
    # (*)
    #print(sum(1 for _ in tf.python_io.tf_record_iterator(trainfiles[0])))
    #print(sum(1 for _ in tf.python_io.tf_record_iterator(valfiles[0])))
    #print(sum(1 for _ in tf.python_io.tf_record_iterator(testfiles[0])))
    #exit() # ...

    tf.reset_default_graph()
    with tf.Session() as sess:

        handle = tf.placeholder(tf.string, shape=[], name="handle_iterator")
        training_handle, train_iter, gen_iter = generator(sess, handle, args["batchsize"], trainfiles, 0)
        val_handle, val_iter, _ = generator(sess, handle, eval_batchsize, valfiles, 1)
        test_handle, test_iter, _ = generator(sess, handle, eval_batchsize, testfiles, 1)

        sample = gen_iter.get_next()
        user_sample = sample[0]
        item_sample = sample[1]
        item_rating = sample[4]

        is_training = tf.placeholder(tf.bool, name="is_training")
        anneal_val = tf.placeholder(tf.float32, name="anneal_val", shape=())
        batch_placeholder = tf.placeholder(tf.int32, name="batch_placeholder")

        model = Model(sample, args)

        item_emb_matrix, item_emb_ph, item_emb_init = model._make_embedding(num_items, args["bits"], "item_embedding")


        user_emb_matrix1, user_emb_ph1, user_emb_init1 = model._make_embedding(num_users, args["bits"], "user_embedding1")
        user_mask_matrix, user_mask_ph, user_mask_init = model._make_embedding(num_users, args["bits"], "user_embedding_mask")


        loss, loss_no_anneal, scores, item_embedding, user_embedding, \
            reconloss, rank_loss_uneq, rank_loss_eq, i1r_m, nonzerobits = model.make_network(item_emb_matrix, user_emb_matrix1, user_mask_matrix, is_training, args, max_rating, anneal_val, batch_placeholder)

        step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(args["lr"],
                                        step,
                                        100000,
                                        args["decay_rate"],
                                        staircase=True, name="lr")

        optimizer = tf.train.AdamOptimizer(learning_rate=args["lr"], name="Adam")
        train_step = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        sess.run(train_iter.initializer)

        eval_list = [loss, scores, user_sample, item_rating, item_sample, item_embedding, user_embedding,  reconloss, rank_loss_uneq, rank_loss_eq]
        counter = 0
        losses_train = []
        losses_train_no_anneal = []
        times = []
        anneal = args["anneal_val"]

        best_val_ndcg = 0
        best_val_loss = np.inf
        patience = 5

        patience_counter = 0
        running = True
        print("starting training")
        all_saves = [args]
        all_val_ndcg = []
        times = []
        while running:

            start = time.time()

            t = time.time()
            lossval, loss_no_anneal_val, hamdist, _, iv, ir, nzb = sess.run([loss, loss_no_anneal, scores, train_step, i1r_m, item_rating, nonzerobits], feed_dict={handle: training_handle, is_training: True,
                                                                                  anneal_val: anneal,
                                                                                  batch_placeholder: args["batchsize"]})
            times.append(time.time() - t)
            times = times[-100:]
            #print(np.mean(times))

            times.append(time.time() - start)
            losses_train.append(lossval)
            losses_train_no_anneal.append(loss_no_anneal_val)
            counter += 1

            anneal = anneal * 0.9999
            if counter % int(1500*args["mul"]) == 0:
                print("train", np.mean(losses_train), np.mean(losses_train_no_anneal), counter * args["batchsize"] / train_samples, np.mean(times), anneal)
                fprint(args["ofile"], " ".join([str(v) for v in ["train", np.mean(losses_train), np.mean(losses_train_no_anneal), counter * args["batchsize"] / train_samples, np.mean(times), anneal]]) )
                losses_train = []
                times = []
                losses_train_no_anneal = []

                sess.run(val_iter.initializer)
                losses_val, val_ndcg, losses_val_recon, losses_val_uneq, losses_val_eq, NN, allndcgs, val_mrr = onepass(sess, eval_list, val_handle,
                                     val_samples, eval_batchsize,
                                     handle, anneal_val, batch_placeholder, is_training, args["force_selfmask"], args)
                print("val ndcg@(5,10) and MRR\t\t", val_ndcg, val_mrr)#, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN)
                save_val_ndcg = val_ndcg
                fprint(args["ofile"], " ".join([str(v) for v in ["val\t\t", val_ndcg, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN]]) )

                all_val_ndcg.append(best_val_ndcg)
                if val_ndcg[-1] > best_val_ndcg:# or best_val_loss > losses_val:
                    best_val_ndcg = val_ndcg[-1]

                    best_val_loss = losses_val
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter == 0:
                    sess.run(test_iter.initializer)
                    _, val_ndcg, losses_val_recon, losses_val_uneq, losses_val_eq, NN, _, val_mrr = onepass(sess, eval_list, test_handle,
                                                   test_samples, eval_batchsize,
                                                   handle, anneal_val, batch_placeholder, is_training, args["force_selfmask"], args)
                    print("test ndcg@(5,10) and MRR\t\t\t\t",  val_ndcg, val_mrr)#, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN)
                    #fprint(args["ofile"], " ".join([str(v) for v in ["test\t\t\t\t",  val_ndcg, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN]]))

                    save_item = sess.run(item_emb_matrix)
                    save_user = sess.run(user_emb_matrix1)

                    to_save = [losses_val, save_val_ndcg, allndcgs, args, all_val_ndcg, save_item, save_user]

                if patience_counter >= patience:
                    running = False


                print("patience", patience_counter, "/", patience, (datetime.datetime.now()))
                fprint(args["ofile"], " ".join([str(v) for v in ["patience", patience_counter, "/", patience, (datetime.datetime.now())]]))

        pickle.dump(to_save, open(savename, "wb"))


if __name__ == "__main__":
    pool = mp.Pool(5)
    main()
    pool.close()
    pool.join()
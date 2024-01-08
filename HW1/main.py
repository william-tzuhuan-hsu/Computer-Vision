from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    # # Namespace(data_dir='../data', feat_dir='../feat', out_dir='.', filter_scales=[1, 2], K=10, alpha=25, L=1)
    opts = get_opts()
    ## Q1.1
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # # shape (375, 500, 3)
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    # # # # test
    # # # alpha = 100
    # # # visual_words.compute_dictionary_one_image(opts, 'kitchen/sun_aasmevtpkslccptd.jpg', img, alpha, 16)

    # # Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # # # Q1.3
    # print("Computing visual words.")
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)

    ## test for Q2.1
    # print(visual_recog.get_feature_from_wordmap(opts, wordmap))

    # test for Q2.2
    # visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)

    # test for Q2.3
    # visual_recog.get_image_feature(opts, img_path, dictionary)

    # Q2.1-2.4
    print("Building recognition system.")
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    print("Inferencing.")
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

if __name__ == '__main__':
    main()

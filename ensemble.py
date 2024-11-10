import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--output-dir',
                        help='Directory to save the fused scores in a pickle file')

    arg = parser.parse_args()
    label = np.load('data/fake_labels.npy')

    fused_scores = {}


    with open('data/CRYD/CRYD-main/com/epoch1_test_score (25).pkl', 'rb') as r1:
        r1 = list(pickle.load(r1).items())
    with open('data/CRYD/CRYD-main/com/epoch1_test_score (26).pkl', 'rb') as r2:
        r2 = list(pickle.load(r2).items())
    with open('data/CRYD/CRYD-main/com/epoch1_test_score (27).pkl', 'rb') as r3:
        r3 = list(pickle.load(r3).items())
    with open('data/CRYD/CRYD-main/com/epoch1_test_score (28).pkl', 'rb') as r4:
        r4 = list(pickle.load(r4).items())
    with open('data/CRYD/CRYD-main/com/epoch1_test_score (29).pkl', 'rb') as r5: #HDBN
        r5 = list(pickle.load(r5).items())
    with open('data/CRYD/CRYD-main/com/epoch1_test_score (30).pkl', 'rb') as r6: #SkateFormer
        r6 = list(pickle.load(r6).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/j/epoch1_test_score.pkl', 'rb') as r7: #DeGCN
        r7 = list(pickle.load(r7).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/ja/epoch1_test_score.pkl', 'rb') as r8: #DeGCN
        r8 = list(pickle.load(r8).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/jm/epoch1_test_score.pkl', 'rb') as r9: #DeGCN
        r9 = list(pickle.load(r9).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/b/epoch1_test_score.pkl', 'rb') as r10: #DeGCN
        r10 = list(pickle.load(r10).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/ba/epoch1_test_score.pkl', 'rb') as r11: #DeGCN
        r11 = list(pickle.load(r11).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/bm/epoch1_test_score.pkl', 'rb') as r12: #DeGCN
        r12 = list(pickle.load(r12).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2/epoch1_test_score.pkl', 'rb') as r13: #DeGCN
        r13 = list(pickle.load(r13).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2a/epoch1_test_score.pkl', 'rb') as r14: #DeGCN
        r14 = list(pickle.load(r14).items())
    with open('data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2M/epoch1_test_score.pkl', 'rb') as r15: #DeGCN
        r15 = list(pickle.load(r15).items())
    with open('data/CRYD/CRYD-main/skate/epoch1_test_score (21).pkl', 'rb') as r16: #DeGCN
        r16 = list(pickle.load(r16).items())
    with open('data/CRYD/CRYD-main/skate/epoch1_test_score (22).pkl', 'rb') as r17: #DeGCN
        r17 = list(pickle.load(r17).items())
    with open('data/CRYD/CRYD-main/skate/jm.pkl', 'rb') as r18: #DeGCN
        r18 = list(pickle.load(r18).items())
    with open('data/CRYD/CRYD-main/skate/epoch1_test_score (23).pkl', 'rb') as r19: #DeGCN
        r19 = list(pickle.load(r19).items())
    with open('data/CRYD/CRYD-main/skate/epoch1_test_score (24).pkl', 'rb') as r20: #DeGCN
        r20 = list(pickle.load(r20).items())
    with open('data/CRYD/CRYD-main/skate/bm.pkl', 'rb') as r21: #DeGCN
        r21 = list(pickle.load(r21).items())
    right_num = total_num = right_num_5 = 0
    fused_scores = {}

    if 1:
        arg.alpha = [20.0, 20.0, 0.0, 20.0, 20.0, 20.0, 20.0, 10.484262434405876, 0.0, 0.0, 13.873746617408498, 13.26193996962492, 20.0, 20.0, 0.0, 13.213218578942596, 20.0, 0.0, 20.0, 20.0, 0.0]
        # arg.alpha = [40.0, 19.15020760818735, 0.1, 40.0, 23.227738429223955, 10.309083953174309, 4.555693027638194, 0.1, 16.339708506864625, 0.1, 3.0981289936224545, 0.1, 13.390323661113518, 0.1, 0.1, 17.615144640880224, 40.0, 0.1, 36.28745529196741, 40.0, 20.083355445321256]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, a11 = r1[i]
            _, a22 = r2[i]
            _, a33 = r3[i]
            _, a44 = r4[i]
            _, a55 = r5[i]
            _, a66 = r6[i]
            _, a77 = r7[i]
            _, a88 = r8[i]
            _, a99 = r9[i]
            _, a1010 = r10[i]
            _, a1111 = r11[i]
            _, a1212 = r12[i]
            _, a1313 = r13[i]
            _, a1414 = r14[i]
            _, a1515 = r15[i]
            _, a1616 = r16[i]
            _, a1717 = r17[i]
            _, a1818 = r18[i]
            _, a1919 = r19[i]
            _, a2020 = r20[i]
            _, a2121 = r21[i]
            r = a11 * arg.alpha[0] + a22 * arg.alpha[1] + a33 * arg.alpha[2] + a44 * arg.alpha[3] + a55 * arg.alpha[4] + a66 * arg.alpha[5] + a77 * arg.alpha[6] + a88 * arg.alpha[7] + a99 * arg.alpha[8] + a1010 * arg.alpha[9] + a1111 * arg.alpha[10] + a1212 * arg.alpha[11] + a1313 * arg.alpha[12] + a1414 * arg.alpha[13] + a1515 * arg.alpha[14] + a1616 * arg.alpha[15] + a1717 * arg.alpha[16] + a1818 * arg.alpha[17] + a1919 * arg.alpha[18] + a2020 * arg.alpha[19] + a2121 * arg.alpha[20] 
            fused_scores[i] = r  # Append the fused score
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num


    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    # Save the fused scores to a pickle file
    if 1:
        output_file = os.path.join('data/CRYD/CRYD-main/com', 'fused_scores.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(fused_scores, f)
        print(f'Fused scores saved to {output_file}')

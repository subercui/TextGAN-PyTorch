# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : visualization.py
# @Time         : Created at 2019-03-19
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import matplotlib.pyplot as plt

title_dict = {
    'gen_pre_loss': 'pre_loss',
    'gen_adv_loss': 'g_loss',
    'gen_mana_loss': 'mana_loss',
    'gen_work_loss': 'work_loss',
    'dis_loss': 'd_loss',
    'dis_train_acc': 'train_acc',
    'dis_eval_acc': 'eval_acc',
    'NLL_oracle': 'NLL_oracle',
    'NLL_gen': 'NLL_gen',
    'BLEU-3': 'BLEU-3',
}

color_list = ['#e74c3c', '#e67e22', '#f1c40f',
              '#8e44ad', '#2980b9', '#27ae60', '#16a085']


def plt_data(data, step, title, c_id, savefig=False):
    x = [i for i in range(step)]
    plt.plot(x, data, color=color_list[c_id], label=title)
    if savefig:
        plt.savefig('savefig/' + title + '.png')


def get_log_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')
        data_dict = {'pre_loss': [], 'g_loss': [], 'mana_loss': [], 'work_loss': [],
                     'd_loss': [], 'train_acc': [], 'eval_acc': [], 'NLL_oracle': [],
                     'NLL_gen': [], 'BLEU-3': []}

        for line in all_lines:
            items = line.split()
            try:
                for key in data_dict.keys():
                    if key == 'BLEU-3' and 'BLEU-[2,' in items:
                        data_dict[key].append(
                            float(items[items.index('BLEU-[2,') + 4][:-2]))
                    if key in items:
                        data_dict[key].append(
                            float(items[items.index(key) + 2][:-1]))
            except:
                break

    return data_dict


if __name__ == '__main__':
    log_file_root = '../log/image_coco/'
    # Custom your log files in lists, no more than len(color_list)
    log_file_list = ['image_coco_s4_RGDv0',
                     'image_coco_s1_RGDv0', 'image_coco_s1_RGDv1', 'image_coco_s1_rel']
    legend_text = ['RGD_T 4-slot', 'RGD_T 1-slot',
                   'RGD_t 1-slot', 'RelGAN 1-slot']

    color_id = 0
    data_name = 'NLL_gen'
    # data_name = 'BLEU-3'
    if_save = False
    # legend_text = log_file_list

    assert data_name in title_dict.keys(), 'Error data name'
    plt.figure(figsize=(6, 8), dpi=300)
    plt.clf()
    plt.title(data_name+' on image_coco')
    all_data_list = []
    for idx, item in enumerate(log_file_list):
        log_file = log_file_root + item + '.txt'

        # save log file
        all_data = get_log_data(log_file)
        plt_data(all_data[title_dict[data_name]], len(all_data[title_dict[data_name]]),
                 legend_text[idx], color_id, if_save)
        color_id += 1

    plt.legend()
    plt.grid()
    plt.xlabel('num of 10 epochs')
    plt.show()

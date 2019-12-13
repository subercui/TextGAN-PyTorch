import json
import os
from os.path import dirname, exists, abspath, join

yelp_data_file = '/home/haotian/Downloads/yelp_dataset/review.json'
yelp_dataset = join(dirname(abspath(__file__)), 'yelp.txt')

if exists(yelp_dataset):
    raise Exception(f'file: {yelp_dataset} already exists')

with open(yelp_data_file, 'r') as f:
    with open(yelp_dataset, 'w') as f_out:
        cnt = 0
        while True:
            # read file
            line = f.readline()
            if not line:
                break
            print(cnt)
            json_content = json.loads(line)

            # parse review
            review = json_content['text']
            review = review.replace('\n', ' ').replace(
                '\t', ' ').replace('  ', ' ').strip()

            # filter length
            if len(review) < 150 or len(review) > 450:
                continue

            # write
            f_out.write(review + '\n')
            # print(review)
            cnt += 1
